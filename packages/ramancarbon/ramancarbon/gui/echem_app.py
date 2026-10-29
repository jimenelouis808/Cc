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

from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from ..echem.curve import POTENTIAL_SCALES, Electrode
from ..echem.cv import SWEEP_CHOICES
from .figures_echem import PRESET_LABELS
from .base import SectionApp, placeholder
from .echem_state import (CURRENT_UNITS, DEFAULT_DISC_MM,
                          DEFAULT_REFERENCE, POTENTIAL_UNITS,
                          EchemSession, _default_electrode)
from .theme import PAD
from .widgets import (
    flow,
    card,
    fill_table,
    hint,
    labelled,
    scrollable_column,
    scrolled_text,
    set_text,
    split_column,
    table,
)


#: Label of the "no choice made" entry in the cycle selector. It is the
#: default, and the selector is rebuilt from the loaded files, so the
#: constant exists only so the entry can be recognised before anything
#: is loaded.
CYCLE_DEFAULT = "último completo"

FILE_TYPES = (
    ("Medidas de potenciostato",
     "*.txt *.csv *.dat *.mpt *.mpr *.DTA *.dta *.asc"),
    ("EC-Lab", "*.mpr *.mpt *.mps"),
    ("Texto", "*.txt *.csv *.dat *.asc"),
    ("Todos", "*.*"),
)


#: Which language the FIGURE labels come out in, as ``(key, label)``.
#:
#: English first and English by default: the window, the report and
#: every message stay Spanish, because those are for the person running
#: the experiment, and the figure is for whoever reads the paper.
LANGUAGE_LABELS: tuple[tuple[str, str], ...] = (
    ("en", "Inglés (para publicar)"),
    ("es", "Español"),
)


def _language_key(label: str) -> str:
    return next((key for key, text in LANGUAGE_LABELS if text == label), "en")


class EchemApp(SectionApp):
    """The electrochemistry section of the suite."""

    def __init__(self, root, container, palette, fonts) -> None:
        super().__init__(root, container, palette, fonts)
        self.session = EchemSession()
        self._drawers = {
            "cv": self._draw_cv,
            "rates": self._draw_rates,
            "gcd": self._draw_gcd,
            "rate_capacitance": self._draw_rate_capacitance,
            "nyquist": self._draw_nyquist,
            "bode": self._draw_bode,
            "drt": self._draw_drt,
            "capacitance": self._draw_capacitance,
            "ragone": self._draw_ragone,
            "tafel": self._draw_tafel,
            "polarisation": self._draw_polarisation,
            "composite": self._draw_composite,
        }
        # By NOTEBOOK INDEX, so it follows the order tabs are ADDED and
        # not the order the methods sit in the file. Inserting a tab in
        # the middle without shifting the rest does not raise: it draws
        # the wrong figure into the wrong tab, or leaves one blank.
        self._tab_canvases = {
            0: (),
            1: ("cv", "rates"),
            2: ("gcd", "rate_capacitance"),
            3: ("nyquist", "bode"),
            4: ("drt",),
            5: ("capacitance", "ragone"),
            6: ("tafel", "polarisation"),
            7: ("composite",),
        }
        self._build()
        self.set_status("Carga una medida o pulsa Demo.")

    # ==================================================================
    # layout
    # ==================================================================
    def _build(self) -> None:
        ttk = self.ttk
        # The status bar FIRST, and from the bottom. It carries the
        # progress line and the elapsed clock, and it was packed after a
        # body with expand=True in the same parent -- so Tk gave the body
        # everything and the bar what was left, which in a real window is
        # nothing. The counter and the clock were computed, queued and
        # drawn to a strip zero pixels tall: a refinement that reports its
        # iteration every ten evaluations looked, from outside, exactly
        # like one that had hung. It is the same packing rule this section
        # has been bitten by before, arriving through the one widget that
        # exists to say the program is alive.
        self.build_status(self.container)
        body = ttk.Frame(self.container, padding=(PAD["md"], PAD["sm"]))
        body.pack(fill="both", expand=True)

        # Scrollable: this column's cards add up to more than any
        # window is tall, and a fixed frame simply clips them.
        sidebar_column, sidebar = scrollable_column(body, width=300)
        sidebar_column.pack(side="left", fill="y", padx=(0, PAD["md"]))
        self._build_sidebar(sidebar)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(side="left", fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._build_tab_summary()
        self._build_tab_cv()
        self._build_tab_gcd()
        self._build_tab_eis()
        self._build_tab_drt()
        self._build_tab_capacitance()
        self._build_tab_catalysis()
        self._build_tab_figure()
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
        self.diameter_var = tk.StringVar(value=f"{DEFAULT_DISC_MM:g}")
        labelled(ebody, "Disco ⌀ (mm)", lambda p: ttk.Entry(
            p, textvariable=self.diameter_var, width=10))
        self.area_var = tk.StringVar(
            value=f"{_default_electrode().area_cm2:.5g}")
        labelled(ebody, "Área (cm²)", lambda p: ttk.Entry(
            p, textvariable=self.area_var, width=10))
        # The diameter is what is written on the electrode; the area is
        # what the normalisation needs. Typing one fills the other, and
        # the area stays editable for anything that is not a disc.
        self.diameter_var.trace_add("write", lambda *_: self._area_from_disc())
        self.volume_var = tk.StringVar(value="")
        labelled(ebody, "Volumen (cm³)", lambda p: ttk.Entry(
            p, textvariable=self.volume_var, width=10))
        self.electrolyte_var = tk.StringVar(
            value=_default_electrode().electrolyte)
        labelled(ebody, "Electrolito", lambda p: ttk.Combobox(
            p, textvariable=self.electrolyte_var, width=14,
            values=_electrolyte_labels()))
        # Picking one fills the pH, which is the only reason the table
        # exists: without a pH there is no RHE scale and no
        # overpotential, and a pH typed from memory is 59 mV per unit of
        # error in everything downstream. The field stays free text for
        # an electrolyte that is not in the table.
        self.electrolyte_var.trace_add("write", lambda *_: self._ph_from_electrolyte())
        self.reference_var = tk.StringVar(value=DEFAULT_REFERENCE)
        labelled(ebody, "Referencia", lambda p: ttk.Combobox(
            p, textvariable=self.reference_var, width=14, state="readonly",
            values=_reference_keys()))
        self.ph_var = tk.StringVar(value=f"{_default_electrode().ph:g}")
        labelled(ebody, "pH", lambda p: ttk.Entry(p, textvariable=self.ph_var,
                                                  width=10))
        self.resistance_var = tk.StringVar(value="")
        labelled(ebody, "R no compensada (Ω)", lambda p: ttk.Entry(
            p, textvariable=self.resistance_var, width=10))
        self.compensated_var = tk.StringVar(value="0")
        labelled(ebody, "Ya compensado (0-1)", lambda p: ttk.Entry(
            p, textvariable=self.compensated_var, width=10))
        ttk.Button(ebody, text="R de la impedancia cargada",
                   command=self._resistance_from_eis).pack(
            fill="x", pady=(PAD["xs"], 0))
        hint(ebody,
             "La masa es la de material ACTIVO, no la del electrodo: el "
             "aglomerante y el carbón conductor suelen ser el 20 %.   "
             "Sin pH no se puede pasar a RHE y no hay sobrepotencial. Sin "
             "resistencia no hay corrección óhmica, y la pendiente de Tafel "
             "sale grande de más. El electrolito no es decoración: la misma "
             "muestra da otra capacitancia en otro, y la ventana útil es del "
             "electrolito, no del material.",
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

        self.potential_unit_var = tk.StringVar(value=POTENTIAL_UNITS[0][0])
        labelled(lbody, "Unidad de E", lambda p: ttk.Combobox(
            p, textvariable=self.potential_unit_var, width=13, state="readonly",
            values=[label for label, _ in POTENTIAL_UNITS]))
        self.current_unit_var = tk.StringVar(value=CURRENT_UNITS[0][0])
        labelled(lbody, "Unidad de I", lambda p: ttk.Combobox(
            p, textvariable=self.current_unit_var, width=13, state="readonly",
            values=[label for label, _ in CURRENT_UNITS]))
        hint(lbody,
             "Sólo para archivos SIN cabecera de unidades; si la trae, gana "
             "la cabecera. No se adivina: una columna de números alrededor de "
             "0.001 puede ser amperios o miliamperios con la misma "
             "verosimilitud, y equivocarse arrastra la corrección óhmica, así "
             "que un sobrepotencial de 170 mV vuelve como 400 V.",
             wrap=260)

        self.scan_rate_var = tk.StringVar(value="20")
        labelled(lbody, "Velocidad (mV/s)", lambda p: ttk.Entry(
            p, textvariable=self.scan_rate_var, width=10))
        self.current_var = tk.StringVar(value="1.0")
        labelled(lbody, "Corriente GCD (mA)", lambda p: ttk.Entry(
            p, textvariable=self.current_var, width=10))
        self.cycle_var = tk.StringVar(value=CYCLE_DEFAULT)
        self.cycle_box = labelled(lbody, "Ciclo a analizar", lambda p: ttk.Combobox(
            p, textvariable=self.cycle_var, width=18, state="readonly",
            values=[CYCLE_DEFAULT]))
        hint(lbody,
             "Por defecto el ÚLTIMO ciclo completo: el primero de cualquier "
             "electrodo lleva lo que sólo pasa una vez, y un ciclo que la "
             "adquisición cortó a medias se integra sobre parte de su "
             "recorrido y da una capacitancia baja en esa proporción. "
             "Elígelo a mano cuando los últimos hayan derivado, o para "
             "comparar varios archivos en el MISMO número de ciclo. La "
             "lista marca los incompletos.",
             wrap=260)

        for text, command in (
            ("Cargar CV…", lambda: self._load("cv")),
            ("Añadir a la serie…", lambda: self._load("rates")),
            ("Cargar GCD…", lambda: self._load("gcd")),
            ("Añadir a la serie GCD…", lambda: self._load("gcd_series")),
            ("Cargar EIS…", lambda: self._load("eis")),
            ("Añadir a la serie de EIS…", lambda: self._load("eis_series")),
            ("Cargar EIS de Nyquist + Bode…", self._load_eis_pair),
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
        self.circuit_box = labelled(obody, "Circuito", lambda p: ttk.Combobox(
            p, textvariable=self.circuit_var, width=16,
            values=self.session.circuit_choices()))
        self.circuit_var.trace_add("write", lambda *_: self._on_circuit_chosen())
        self.non_faradaic_var = tk.BooleanVar(value=False)
        # A checkbutton's text does not wrap on its own, so a sentence in
        # one is a sentence that gets cut: this one was given 276 px of
        # the 350 it needs. Short label, explanation underneath.
        ttk.Checkbutton(
            obody, text="Sin corriente faradaica",
            variable=self.non_faradaic_var,
        ).pack(anchor="w", pady=(PAD["xs"], 0))
        hint(obody, "Marca esto si la ventana del CV no tiene picos redox: "
                    "la capacitancia se integra entonces sobre toda la "
                    "ventana en vez de descontarlos.", wrap=420)

        self.scale_var = tk.StringVar(value=POTENTIAL_SCALES[0][1])
        labelled(obody, "Eje de potencial", lambda p: ttk.Combobox(
            p, textvariable=self.scale_var, width=22, state="readonly",
            values=[label for _, label in POTENTIAL_SCALES]))
        hint(obody,
             "El eje se dibuja y se exporta COMO SE MIDIÓ salvo que elijas "
             "otra cosa. Una conversión es una afirmación —a RHE son 59 mV "
             "por unidad de pH, y entre dos referencias acuosas es el relleno "
             "que de verdad tuviera la tuya— y lo medido no afirma nada. Lo "
             "que se haya convertido va escrito en la etiqueta del eje. El "
             "Tafel es la excepción: un sobrepotencial se define frente a "
             "RHE, así que esa figura convierte siempre.",
             wrap=420)

        self.dunn_nonneg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(obody, text="Dunn: coeficientes ≥ 0",
                        variable=self.dunn_nonneg_var).pack(
            anchor="w", pady=(PAD["xs"], 0))
        hint(obody,
             "Ninguna de las dos contribuciones puede restar. Sin la "
             "restricción, una k₁ negativa dibuja la curva capacitiva al "
             "otro lado del lazo. Quítala sólo para ver el ajuste libre.",
             wrap=420)
        self.dunn_points_var = tk.StringVar(value="60")
        labelled(obody, "Dunn: nº de potenciales", lambda p: ttk.Entry(
            p, textvariable=self.dunn_points_var, width=10))
        self.dunn_low_var = tk.StringVar(value="")
        labelled(obody, "Dunn: E mínimo (V)", lambda p: ttk.Entry(
            p, textvariable=self.dunn_low_var, width=10))
        self.dunn_high_var = tk.StringVar(value="")
        labelled(obody, "Dunn: E máximo (V)", lambda p: ttk.Entry(
            p, textvariable=self.dunn_high_var, width=10))
        hint(obody,
             "Vacíos, Dunn usa la ventana común a todas las curvas, que es "
             "el único defecto honrado: un potencial que una de ellas no "
             "alcanzó no puede entrar en un ajuste entre curvas.", wrap=420)

        self.cap_low_var = tk.StringVar(value="")
        labelled(obody, "C: ventana mínima (V)", lambda p: ttk.Entry(
            p, textvariable=self.cap_low_var, width=10))
        self.cap_high_var = tk.StringVar(value="")
        labelled(obody, "C: ventana máxima (V)", lambda p: ttk.Entry(
            p, textvariable=self.cap_high_var, width=10))
        hint(obody,
             "Cambia el DIVISOR de la capacitancia, no la región que se "
             "integra: sirve para citar un valor sobre una ventana "
             "declarada, no para dejar fuera un trozo del barrido.",
             wrap=420)

        self.subtract_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(obody, text="Restar la corriente parásita",
                        variable=self.subtract_var).pack(
            anchor="w", pady=(PAD["xs"], 0))
        hint(obody,
             "Quita de la serie de velocidades la corriente constante que "
             "mide el BALANCE DE CARGA (ΔQ contra la duración del ciclo), "
             "que no mira cómo escala i con ν y por eso no es circular. "
             "Sirve para ver si el b vuelve a 1 sin la fuga. Apagado por "
             "defecto: una corrección encendida por defecto deja de verse.",
             wrap=420)

        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=(PAD["sm"], 0))
        ttk.Button(actions, text="Analizar", style="Accent.TButton",
                   command=self._analyse).pack(fill="x", pady=(0, PAD["xs"]))
        ttk.Button(actions, text="Demo", command=self._load_demo).pack(
            fill="x", pady=(0, PAD["xs"]))
        ttk.Button(actions, text="Guardar informe…",
                   command=self._save_report).pack(fill="x",
                                                   pady=(0, PAD["xs"]))
        ttk.Button(actions, text="Exportar tablas…",
                   command=self._export_tables).pack(fill="x")

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

        dunn_bar = flow(tab, style="TFrame")
        dunn_bar.add(ttk.Label(dunn_bar.frame, text="Dunn — rama:"))
        self.dunn_sweep_var = self.tk.StringVar(value=SWEEP_CHOICES[0][1])
        dunn_bar.add(ttk.Combobox(
            dunn_bar.frame, textvariable=self.dunn_sweep_var, width=38,
            state="readonly",
            values=[label for _, label in SWEEP_CHOICES]), grow=True)
        dunn_bar.add(ttk.Label(dunn_bar.frame,
                               text="velocidad del gráfico (mV/s):"))
        self.dunn_rate_var = self.tk.StringVar(value="")
        self.dunn_rate_box = ttk.Combobox(
            dunn_bar.frame, textvariable=self.dunn_rate_var, width=10,
            state="readonly", values=[])
        dunn_bar.add(self.dunn_rate_box)
        self.dunn_rate_box.bind("<<ComboboxSelected>>",
                                lambda _e: self._on_dunn_rate())
        dunn_bar.add(ttk.Button(dunn_bar.frame, text="Recalcular Dunn",
                                command=self._recompute_dunn))
        # Under the row, not inside it: a paragraph packed beside three
        # controls competes with them for width and loses -- this one was
        # given 121 px of the 520 it asked for, on a 1480 px window.
        hint(tab,
             "El gráfico se dibuja a la velocidad MÁS LENTA por defecto, "
             "que es donde la contribución difusiva es mayor: a la más "
             "rápida todo electrodo parece superficial.",
             wrap=560)
        paned, (top, bottom) = split_column(tab, (3, 2))
        paned.pack(fill="both", expand=True)
        outer, body = card(top, None)
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "cv", lambda f: f.add_subplot(111))
        rates_card, rates_body = card(bottom, None)
        rates_card.pack(fill="both", expand=True)
        self.make_canvas(rates_body, "rates",
                         lambda f: (f.add_subplot(131), f.add_subplot(132),
                                    f.add_subplot(133)),
                         figsize=(7.6, 3.4))

    def _build_tab_gcd(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Carga-descarga  ")
        paned, (top, middle, bottom) = split_column(tab, (3, 2, 2))
        paned.pack(fill="both", expand=True)
        outer, body = card(top, None)
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "gcd", lambda f: f.add_subplot(111))

        rate_card, rate_body = card(middle, "Capacitancia y capacidad de velocidad")
        rate_card.pack(fill="both", expand=True)
        inner = ttk.Frame(rate_body)
        inner.pack(fill="both", expand=True)
        left = ttk.Frame(inner)
        left.pack(side="left", fill="both", expand=True)
        self.gcd_capacitance_table = table(
            left, ["convenio", "C (mF)", "C (F/g)"], height=5)
        hint(left,
             "Tres convenios, no uno. En una descarga recta coinciden en un "
             "1 %; en una meseta difieren un 30 %, y el de ΔV sobreinforma "
             "exactamente lo que la curva se dobla porque supone que es una "
             "recta. Para un pseudocondensador el defendible es el de "
             "ENERGÍA: la capacitancia que almacenaría la misma energía en "
             "la misma ventana.",
             wrap=420)
        right = ttk.Frame(inner)
        right.pack(side="left", fill="both", expand=True, padx=(PAD["sm"], 0))
        self.make_canvas(right, "rate_capacitance",
                         lambda f: f.add_subplot(111), figsize=(4.4, 2.8))

        info, info_body = card(bottom, "Ramas")
        info.pack(fill="both", expand=True)
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
        paned, (top, middle, lower) = split_column(tab, (3, 2, 3))
        paned.pack(fill="both", expand=True)
        outer, body = card(top, None)
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "nyquist", lambda f: f.add_subplot(111),
                         figsize=(6.0, 5.0))
        bode_card, bode_body = card(middle, None)
        bode_card.pack(fill="both", expand=True)
        self.make_canvas(bode_body, "bode",
                         lambda f: (f.add_subplot(121), f.add_subplot(122)),
                         figsize=(7.6, 3.4))
        # The circuit editor and its two tables get their own pane with a
        # scrollbar rather than being packed under the figures, where the
        # figures take the height first and leave them a sliver.
        controls_column, controls = scrollable_column(lower, width=640)
        controls_column.pack(fill="both", expand=True)
        editor, editor_body = card(controls, "Circuito equivalente")
        editor.pack(fill="x")
        row = ttk.Frame(editor_body)
        row.pack(fill="x")
        ttk.Label(row, text="Circuito").pack(side="left")
        self.circuit_text_var = self.tk.StringVar(
            value=self.session.circuit_text())
        # A wrapping row: four controls side by side in this panel left
        # "Guardar como…" 52 px of the 145 it needs.
        bar = flow(row, style="TFrame")
        bar.add(ttk.Entry(bar.frame, textvariable=self.circuit_text_var,
                          width=28), grow=True)
        bar.add(ttk.Button(bar.frame, text="Aplicar",
                           command=self._apply_circuit))
        # Applying a circuit used to update the tables and then do
        # nothing: the fit only happened when the whole analysis was
        # re-run from the sidebar, which also redid the CV, the GCD, Dunn
        # and the catalysis. Choosing a circuit is a loop — write one,
        # look at the residual, write another — and it costs one fit.
        bar.add(ttk.Button(bar.frame, text="Ajustar",
                           command=self._refit_circuit))
        bar.add(ttk.Button(bar.frame, text="Dibujar",
                           command=self._draw_circuit))
        bar.add(ttk.Button(bar.frame, text="Comparar…",
                           command=self._compare_circuits))
        bar.add(ttk.Button(bar.frame, text="Guardar como…",
                           command=self._save_circuit))
        bar.add(ttk.Button(bar.frame, text="Borrar el mío",
                           command=self._delete_circuit))
        self.circuit_note = ttk.Label(
            editor_body, text="", wraplength=900, justify="left",
            style="Hint.TLabel")
        self.circuit_note.pack(fill="x", pady=(PAD["xs"], 0))
        hint(editor_body,
             "- serie, | paralelo, paréntesis agrupan. Elementos: R "
             "resistencia, C condensador, L inductancia, La inductancia "
             "no ideal, Q CPE, Za ZARC (R‖CPE en R, τ y γ), W Warburg "
             "semiinfinito, Ws finito transmisivo, Wo finito reflectante, "
             "G Gerischer, T electrodo poroso (línea de transmisión). "
             "Ejemplo: R0-(R1|Q1)-T1.  OJO: «|» liga MÁS fuerte que «-», "
             "así que R1|Q1|R2-L1 es (R1|Q1|R2)-L1 y no R1|Q1|(R2-L1). "
             "Para una rama que sea ella misma una serie, ponle "
             "paréntesis — y púlsale a «Dibujar», que es donde se ve.",
             wrap=900)
        self.circuit_drawing = self.tk.Text(
            editor_body, height=7, wrap="none", font=("TkFixedFont", 9))
        self.circuit_drawing.pack(fill="x", pady=(PAD["xs"], 0))
        self.circuit_drawing.configure(state="disabled")

        setup, setup_body = card(controls, "Valores de partida y parámetros fijos")
        setup.pack(fill="x", pady=(PAD["sm"], 0))
        self.circuit_setup_table = table(
            setup_body, ["parámetro", "valor de partida", "fijo"], height=6)
        self.circuit_setup_table.bind("<Double-1>", self._edit_circuit_parameter)
        buttons = ttk.Frame(setup_body)
        buttons.pack(fill="x", pady=(PAD["xs"], 0))
        ttk.Button(buttons, text="Fijar / soltar",
                   command=self._toggle_circuit_fixed).pack(side="left")
        ttk.Button(buttons, text="Restablecer",
                   command=self._reset_circuit_parameters).pack(
                       side="left", padx=(PAD["xs"], 0))
        hint(setup_body,
             "Doble clic para cambiar un valor de partida. Fijar un "
             "parámetro NO es gratis: deja de contar como grado de libertad, "
             "así que TODAS las demás incertidumbres salen más pequeñas, y si "
             "el valor fijado está mal el error se reparte entre sus vecinos "
             "sin que el ajuste empeore. Es lo correcto para lo que la medida "
             "no determina — la inductancia de los cables si paraste en "
             "100 kHz — y un engaño para todo lo demás.",
             wrap=900)

        info, info_body = card(controls, "Parámetros del circuito")
        info.pack(fill="x", pady=(PAD["sm"], 0))
        self.circuit_table = table(
            info_body, ["parámetro", "valor", "incertidumbre", "fijo"],
            height=7)
        exports = ttk.Frame(info_body)
        exports.pack(fill="x", pady=(PAD["xs"], 0))
        ttk.Button(exports, text="Exportar la curva ajustada…",
                   command=self._export_fit_curve).pack(side="left")
        ttk.Button(exports, text="Capacitancia efectiva…",
                   command=self._effective_capacitance).pack(
                       side="left", padx=(PAD["xs"], 0))
        hint(info_body,
             "Kramers-Kronig va ANTES que el circuito: un ajuste a datos que "
             "han derivado da parámetros sin significado y el χ² no lo delata. "
             "Y la Q de un CPE no son faradios.",
             wrap=900)

    def _build_tab_drt(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  DRT  ")
        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=(0, PAD["sm"]))
        self.drt_auto_var = self.tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls, text="Elegir λ por la curva L",
            variable=self.drt_auto_var, command=self._on_drt_auto,
        ).pack(side="left")
        ttk.Label(controls, text="λ").pack(side="left", padx=(PAD["md"], 0))
        self.drt_lambda_var = self.tk.StringVar(value="")
        self.drt_lambda_entry = ttk.Entry(
            controls, textvariable=self.drt_lambda_var, width=12,
            state="disabled")
        self.drt_lambda_entry.pack(side="left", padx=(PAD["xs"], PAD["md"]))
        ttk.Button(controls, text="Recalcular",
                   command=self._recompute_drt).pack(side="left")

        paned, (top, bottom) = split_column(tab, (3, 2))
        paned.pack(fill="both", expand=True)
        outer, body = card(top, None)
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "drt", lambda f: f.add_subplot(111),
                         figsize=(7.6, 4.2))
        info, info_body = card(bottom, "Procesos resueltos")
        info.pack(fill="both", expand=True)
        self.drt_table = table(
            info_body, ["τ (s)", "R (Ω)", "C = τ/R (mF)", "fracción de R"],
            height=6)
        hint(info_body,
             "La DRT es una inversión MAL CONDICIONADA: λ no es un detalle "
             "de implementación, es una elección sobre cuánta estructura "
             "creerse, y una década a cada lado del codo de la curva L da una "
             "DRT igual de defendible con otro número de picos. γ está "
             "penalizada en su RUGOSIDAD, no en su tamaño, así que el fallo "
             "honrado es fundir dos constantes de tiempo vecinas — «éstas dos "
             "no se resuelven» — y no encoger las dos resistencias. Y dos picos "
             "a menos de un factor de tres en τ son uno.",
             wrap=900)

    def _build_tab_capacitance(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Capacitancia  ")
        paned, (top, bottom) = split_column(tab, (1, 1))
        paned.pack(fill="both", expand=True)
        outer, body = card(top, "C(ω) sin ajustar nada")
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "capacitance",
                         lambda f: (f.add_subplot(121), f.add_subplot(122)),
                         figsize=(7.6, 3.4))
        hint(body,
             "Un circuito es una hipótesis; C(ω) = 1/(jωZ) son los datos. El "
             "máximo de C″ da τ₀ sin necesitar la masa. Ojo: aquí Z″ se "
             "guarda con su signo físico, así que código escrito contra el "
             "−Z″ de un Nyquist devuelve capacitancias negativas.",
             wrap=900)

        compare_card, compare_body = card(bottom, "La misma muestra por tres métodos")
        compare_card.pack(fill="x")
        self.capacitance_table = table(
            compare_body, ["método", "condición", "C (mF)", "C (F/g)"],
            height=5)
        hint(compare_body,
             "No son la misma medida. Lo que entrega un dispositivo es la de "
             "GCD; la de EIS se mide con 10 mV alrededor de un punto fijo, "
             "donde nada está limitado por velocidad, y es una COTA SUPERIOR "
             "que el dispositivo nunca ve. Una dispersión por encima del 30 % "
             "ES el resultado.",
             wrap=900)

        ragone_card, ragone_body = card(bottom, "Ragone")
        ragone_card.pack(fill="both", expand=True, pady=(PAD["sm"], 0))
        self.make_canvas(ragone_body, "ragone", lambda f: f.add_subplot(111),
                         figsize=(6.0, 3.6))
        hint(ragone_body,
             "Un punto por curva de carga-descarga: carga varias a "
             "corrientes distintas con «Añadir a la serie GCD…», porque todo "
             "el contenido de la figura es cómo cae la energía al subir la "
             "potencia. Y un Ragone sin decir su base no se compara con nada: "
             "por gramo de material activo y por kilogramo de celda "
             "empaquetada difieren en un factor de tres a cinco. Éstos son por "
             "gramo de material activo.",
             wrap=900)

    def _build_tab_catalysis(self) -> None:
        ttk = self.ttk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  HER / OER  ")
        paned, (top, middle, bottom) = split_column(tab, (3, 3, 2))
        paned.pack(fill="both", expand=True)
        outer, body = card(top, "Tafel")
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "tafel", lambda f: f.add_subplot(111))
        pol_card, pol_body = card(middle, "Curva de polarización")
        pol_card.pack(fill="both", expand=True)
        self.make_canvas(pol_body, "polarisation", lambda f: f.add_subplot(111),
                         figsize=(7.6, 3.2))
        hint(pol_body,
             "El gráfico de Tafel es un logaritmo y esconde la FORMA. Una "
             "corriente que deja de subir es transporte de materia o una "
             "película de burbujas, y en escala logarítmica parece un cambio "
             "de pendiente que se informa como una segunda región de Tafel.",
             wrap=820)
        info, info_body = card(bottom, "Actividad")
        info.pack(fill="both", expand=True)
        self.catalysis_text = scrolled_text(info_body, self.palette,
                                            self.fonts["mono"], height=12)

    def _build_tab_figure(self) -> None:
        """The multi-panel figure: choose the panels, lay them out, export.

        It exists because the project's published figures were written by
        hand, one matplotlib script per analysis, which puts them out of
        reach of anyone who does not write matplotlib and lets them drift
        from what the window shows. Every panel here draws from this same
        session.
        """
        ttk, tk = self.ttk, self.tk
        tab = ttk.Frame(self.notebook, padding=PAD["md"])
        self.notebook.add(tab, text="  Figura  ")

        # The chooser on the left in its own scrollable column: the
        # panel list grows with the package and a fixed frame clips it.
        chooser_column, chooser = scrollable_column(tab, width=330)
        chooser_column.pack(side="left", fill="y", padx=(0, PAD["md"]))

        quick_card, quick_body = card(chooser, "Empezar")
        quick_card.pack(fill="x")
        ttk.Button(quick_body, text="Figura estándar",
                   command=self._standard_figure).pack(fill="x")
        self.figure_preset_var = tk.StringVar(value=PRESET_LABELS[0][1])
        labelled(quick_body, "Para", lambda p: ttk.Combobox(
            p, textvariable=self.figure_preset_var, width=22,
            state="readonly",
            values=[label for _, label in PRESET_LABELS]))
        self.figure_preset_var.trace_add("write",
                                         lambda *_: self._apply_preset())
        self.figure_language_var = tk.StringVar(value=LANGUAGE_LABELS[0][1])
        labelled(quick_body, "Etiquetas", lambda p: ttk.Combobox(
            p, textvariable=self.figure_language_var, width=22,
            state="readonly",
            values=[label for _, label in LANGUAGE_LABELS]))
        self.figure_language_var.trace_add("write",
                                           lambda *_: self._refresh_composite())
        hint(quick_body,
             "«Figura estándar» elige los paneles que este campo publica "
             "para lo que tengas cargado, que es una acción en vez de una "
             "columna de casillas. Las etiquetas de los EJES salen en "
             "inglés por defecto; la ventana, el informe y los avisos "
             "siguen en español — son dos públicos distintos.")

        panels_card, panels_body = card(
            chooser, "Paneles",
            "sólo se ofrecen los que tienen datos cargados")
        panels_card.pack(fill="x", pady=(PAD["sm"], 0))
        self.figure_panel_vars: dict[str, object] = {}
        self.figure_panels_frame = ttk.Frame(panels_body)
        self.figure_panels_frame.pack(fill="x")

        layout_card, layout_body = card(chooser, "Disposición")
        layout_card.pack(fill="x", pady=(PAD["sm"], 0))
        self.figure_columns_var = tk.StringVar(value="2")
        labelled(layout_body, "Columnas", lambda p: ttk.Spinbox(
            p, from_=1, to=4, width=6, textvariable=self.figure_columns_var))
        self.figure_width_var = tk.StringVar(value="7.2")
        labelled(layout_body, "Ancho (pulgadas)", lambda p: ttk.Entry(
            p, textvariable=self.figure_width_var, width=8))
        self.figure_height_var = tk.StringVar(value="3.0")
        labelled(layout_body, "Alto por panel", lambda p: ttk.Entry(
            p, textvariable=self.figure_height_var, width=8))
        self.figure_font_var = tk.StringVar(value="8.5")
        labelled(layout_body, "Tamaño de letra", lambda p: ttk.Entry(
            p, textvariable=self.figure_font_var, width=8))
        self.figure_dpi_var = tk.StringVar(value="600")
        labelled(layout_body, "ppp al exportar", lambda p: ttk.Entry(
            p, textvariable=self.figure_dpi_var, width=8))
        self.figure_title_var = tk.StringVar(value="")
        labelled(layout_body, "Título", lambda p: ttk.Entry(
            p, textvariable=self.figure_title_var, width=16))
        hint(layout_body,
             "El ancho va emparejado con el tamaño de letra: una columna "
             "de revista mide unas 3.3 pulgadas y dos unas 7, y una figura "
             "dibujada a 7 y encogida a 3.3 llega con todas las etiquetas "
             "a la mitad. Por eso se exporta a ESTE ancho y no se guarda "
             "el lienzo de la pantalla, que mide lo que mida el panel.")

        options_card, options_body = card(chooser, "Opciones")
        options_card.pack(fill="x", pady=(PAD["sm"], 0))
        self.figure_letters_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_body, text="Letras (a), (b), (c)…",
                        variable=self.figure_letters_var,
                        command=self._refresh_composite).pack(anchor="w")
        self.figure_inside_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_body, text="…dentro del marco",
                        variable=self.figure_inside_var,
                        command=self._refresh_composite).pack(anchor="w",
                                                              padx=(18, 0))
        self.figure_legends_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_body, text="Leyendas",
                        variable=self.figure_legends_var,
                        command=self._refresh_composite).pack(anchor="w")
        self.figure_reference_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_body, text="Eje RHE y línea del O₂",
                        variable=self.figure_reference_var,
                        command=self._refresh_composite).pack(anchor="w")
        self.figure_grid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_body, text="Rejilla",
                        variable=self.figure_grid_var,
                        command=self._refresh_composite).pack(anchor="w")

        actions_card, actions_body = card(chooser, "Exportar")
        actions_card.pack(fill="x", pady=(PAD["sm"], 0))
        bar = flow(actions_body)
        bar.add(ttk.Button(bar.frame, text="Redibujar",
                           command=self._refresh_composite))
        bar.add(ttk.Button(bar.frame, text="Guardar figura…",
                           command=self._save_composite))
        bar.add(ttk.Button(bar.frame, text="Guardar los datos…",
                           command=self._save_composite_data))
        hint(actions_body,
             "«Guardar figura» escribe PNG o PDF al ancho y a los puntos "
             "por pulgada de arriba. «Guardar los datos» escribe un CSV "
             "por panel con los números DIBUJADOS, con todo lo que la "
             "figura les haya hecho ya aplicado: una figura que sólo se "
             "puede mirar hay que teclearla para usarla en otro sitio, y "
             "los números tecleados son números equivocados. Un archivo "
             "por panel porque los paneles no comparten abscisa.")

        outer, body = card(tab, None)
        outer.pack(fill="both", expand=True)
        self.make_canvas(body, "composite", lambda f: f.add_subplot(111),
                         figsize=(7.6, 7.2))
        self._refresh_figure_panels()

    def _refresh_figure_panels(self) -> None:
        """Rebuild the checkbox list from what is loaded."""
        for child in self.figure_panels_frame.winfo_children():
            child.destroy()
        self.figure_panel_vars = {}
        panels = self.session.figure_panels()
        if not panels:
            self.ttk.Label(
                self.figure_panels_frame,
                text="Carga una serie de velocidades o un espectro de "
                     "impedancia.",
                wraplength=280, justify="left").pack(anchor="w")
            return
        chosen = set(self.session.composite.panels)
        for panel in panels:
            variable = self.tk.BooleanVar(value=panel.key in chosen)
            self.figure_panel_vars[panel.key] = variable
            # Ticking a box redraws. Having to find a button afterwards
            # turns choosing panels — which is a loop, tick one, look,
            # tick another — into a chore nobody finishes.
            self.ttk.Checkbutton(
                self.figure_panels_frame, text=panel.label,
                variable=variable,
                command=self._refresh_composite).pack(anchor="w")
            if panel.note:
                self.ttk.Label(
                    self.figure_panels_frame, text=panel.note,
                    wraplength=276, justify="left",
                    style="Hint.TLabel").pack(anchor="w", padx=(22, 0))

    def _composite_from_widgets(self) -> None:
        def number(variable, fallback):
            try:
                return float(variable.get())
            except (TypeError, ValueError):
                return fallback

        composite = self.session.composite
        self.session.set_figure_panels(
            [key for key, var in self.figure_panel_vars.items() if var.get()])
        composite.columns = max(1, int(number(self.figure_columns_var, 2)))
        composite.width_in = max(2.0, number(self.figure_width_var, 7.2))
        composite.panel_height_in = max(1.5,
                                        number(self.figure_height_var, 3.0))
        composite.font_size = max(4.0, number(self.figure_font_var, 8.5))
        composite.dpi = max(72, int(number(self.figure_dpi_var, 600)))
        composite.language = _language_key(self.figure_language_var.get())
        composite.letters = bool(self.figure_letters_var.get())
        composite.letters_inside = bool(self.figure_inside_var.get())
        composite.legends = bool(self.figure_legends_var.get())
        composite.reference_axis = bool(self.figure_reference_var.get())
        composite.grid = bool(self.figure_grid_var.get())
        composite.title = self.figure_title_var.get().strip()

    def _composite_to_widgets(self) -> None:
        """Put the layout fields back on screen after a preset moved them."""
        composite = self.session.composite
        self.figure_columns_var.set(str(composite.columns))
        self.figure_width_var.set(f"{composite.width_in:g}")
        self.figure_height_var.set(f"{composite.panel_height_in:g}")
        self.figure_font_var.set(f"{composite.font_size:g}")
        self.figure_dpi_var.set(str(composite.dpi))

    def _apply_preset(self) -> None:
        label = self.figure_preset_var.get()
        key = next((k for k, text in PRESET_LABELS if text == label), "")
        if not key or not self.session.use_figure_preset(key):
            return
        self._composite_to_widgets()
        self._refresh_composite()

    def _standard_figure(self) -> None:
        """Choose the panels this field publishes for what is loaded."""
        chosen = set(self.session.standard_figure())
        for key, variable in self.figure_panel_vars.items():
            variable.set(key in chosen)
        self.flush_messages(self.session.messages)
        if chosen:
            self._refresh_composite()

    def _refresh_composite(self) -> None:
        self._composite_from_widgets()
        self.with_style("composite", self._draw_composite)
        self.flush_messages(self.session.messages)
        count = len(self.session.composite.panels)
        self.set_status(
            f"figura de {count} panel(es), "
            f"{self.session.composite.rows}×{self.session.composite.columns}"
            if count else "elige al menos un panel")

    def _draw_composite(self, figure) -> None:
        self.session.draw_figure(figure, self.figure_palette)

    def _save_composite(self) -> None:
        from tkinter import filedialog

        self._composite_from_widgets()
        if not self.session.composite.panels:
            self.warn("Figura vacía", "Elige al menos un panel.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar la figura", defaultextension=".png",
            filetypes=(("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"),
                       ("Todos", "*.*")),
            parent=self.root)
        if not path:
            return
        saved = self.session.save_figure(path, self.figure_palette)
        self.flush_messages(self.session.messages)
        if saved:
            self.set_status(f"figura guardada en {Path(path).name}")

    def _save_composite_data(self) -> None:
        from tkinter import filedialog

        self._composite_from_widgets()
        if not self.session.composite.panels:
            self.warn("Figura vacía", "Elige al menos un panel.")
            return
        folder = filedialog.askdirectory(
            title="Carpeta para los datos de la figura", parent=self.root)
        if not folder:
            return
        written = self.session.save_figure_data(folder)
        self.flush_messages(self.session.messages)
        if written:
            self.set_status("escritos: " + ", ".join(written))

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
            volume_cm3=number(self.volume_var),
            reference=self.reference_var.get(),
            ph=number(self.ph_var),
            resistance_ohm=number(self.resistance_var),
            ir_compensated_fraction=number(self.compensated_var) or 0.0,
            electrolyte=self.electrolyte_var.get().strip(),
            label=self.name_var.get(),
        )

    def _resistance_from_eis(self) -> None:
        """Take R_u off the impedance of THIS cell and use it everywhere.

        The same cell is the point: a resistance from another day, another
        electrolyte level or another contact is a number with the right
        units and no meaning.
        """
        from ..echem.eis import uncompensated_resistance

        if self.session.eis is None:
            self.warn("Sin impedancia",
                      "Carga un espectro de impedancia de esta misma celda.")
            return
        value, how = uncompensated_resistance(self.session.eis)
        self.resistance_var.set(f"{value:.4g}")
        self._settings_from_widgets()
        self.set_status(f"R_u = {value:.4g} Ω — {how}")

    # -- the Dunn controls ---------------------------------------------
    def _on_dunn_rate(self) -> None:
        value = _number(self.dunn_rate_var.get())
        self.session.dunn_rate = value / 1000.0 if value else None
        self.mark_dirty("rates")
        self.flush_dirty(self._visible())

    def _recompute_dunn(self) -> None:
        """Re-run the separation on the chosen branch, nothing else."""
        from ..echem.cv import dunn_analysis

        labels = {label: key for key, label in SWEEP_CHOICES}
        self.session.dunn_sweep = labels.get(self.dunn_sweep_var.get(), "media")
        if len(self.session.rate_series) < 3:
            self.warn("Hacen falta tres velocidades",
                      "La separación ajusta dos coeficientes por potencial. "
                      "Con dos, el ajuste pasa exactamente por los dos puntos "
                      "y el reparto es el que quieras.")
            return
        if self.session.result is None:
            self._analyse()
            return
        try:
            self.session.result.dunn = dunn_analysis(
                self.session.rate_series, sweep=self.session.dunn_sweep)
        except Exception as error:               # noqa: BLE001 - shown to user
            self.warn("Dunn", str(error))
            return
        self._refresh_dunn_rates()
        self._fill_tables()
        self.mark_dirty("rates")
        self.flush_dirty(self._visible())
        self.set_status(
            f"Dunn recalculado — {dict(SWEEP_CHOICES)[self.session.dunn_sweep]}")

    def _refresh_dunn_rates(self) -> None:
        rates = self.session.dunn_rate_choices()
        self.dunn_rate_box.configure(
            values=[f"{1e3 * rate:g}" for rate in rates])
        chosen = self.session.dunn_rate_for_plot()
        self.dunn_rate_var.set(f"{1e3 * chosen:g}" if chosen else "")

    # -- the DRT controls ----------------------------------------------
    def _on_drt_auto(self) -> None:
        automatic = bool(self.drt_auto_var.get())
        self.drt_lambda_entry.configure(
            state="disabled" if automatic else "normal")
        if automatic:
            self.session.drt_regularisation = None
            self.drt_lambda_var.set("")
        elif self.session.result is not None and self.session.result.drt:
            # Start from the one the L-curve chose rather than from
            # nothing: a lambda typed blind is a lambda off by decades.
            self.drt_lambda_var.set(
                f"{self.session.result.drt.regularisation:.4g}")

    def _recompute_drt(self) -> None:
        """Re-solve the DRT alone, without redoing the whole analysis."""
        from ..echem.eis import drt

        if self.session.eis is None:
            self.warn("Sin impedancia",
                      "Carga un espectro de impedancia primero.")
            return
        if self.drt_auto_var.get():
            self.session.drt_regularisation = None
        else:
            value = _number(self.drt_lambda_var.get())
            if value is None or value <= 0:
                self.warn("λ no válida",
                          "λ tiene que ser un número positivo, o marca la "
                          "curva L para que lo elija.")
                return
            self.session.drt_regularisation = value
        if self.session.result is None:
            self._analyse()
            return
        self.session.result.drt = drt(
            self.session.eis, regularisation=self.session.drt_regularisation)
        self._fill_tables()
        self.mark_dirty("drt")
        self.flush_dirty(self._visible())
        self.set_status(self.session.result.drt.describe())

    # -- the circuit editor --------------------------------------------
    def _on_circuit_chosen(self) -> None:
        """A name picked from the list fills the box with its string.

        A preset the user can then edit is more useful than one they can
        only accept, and it is how they learn the notation.
        """
        name = self.circuit_var.get()
        if not name or not hasattr(self, "circuit_text_var"):
            return
        if not self.session.set_circuit(name):
            self.flush_messages(self.session.messages)
            return
        self.circuit_text_var.set(self.session.circuit_text())
        self._show_circuit_note()
        self._fill_circuit_setup()
        self._refresh_dunn_rates()
        fill_table(self.gcd_capacitance_table,
                   ["convenio", "C (mF)", "C (F/g)"],
                   self.session.gcd_capacitance_rows())
        fill_table(self.drt_table,
                   ["τ (s)", "R (Ω)", "C = τ/R (mF)", "fracción de R"],
                   self.session.drt_rows())
        fill_table(self.capacitance_table,
                   ["método", "condición", "C (mF)", "C (F/g)"],
                   self.session.capacitance_rows())

    def _show_circuit_note(self) -> None:
        use, caution = self.session.circuit_note()
        text = use
        if caution:
            text = f"{use}   ⚠ {caution}" if use else f"⚠ {caution}"
        self.circuit_note.configure(text=text)

    def _apply_circuit(self) -> None:
        """Take whatever is in the box, whether it is a name or a circuit."""
        if not self.session.set_circuit(self.circuit_text_var.get()):
            self.flush_messages(self.session.messages)
            return
        self.circuit_text_var.set(self.session.circuit_text())
        self._show_circuit_note()
        self._fill_circuit_setup()
        self._draw_circuit()
        self.set_status(f"circuito: {self.session.circuit_text()}")

    def _refit_circuit(self) -> None:
        """Apply whatever is in the box and fit it, in one press.

        Applying and fitting were two steps and the second one lived on
        the other side of the window, so changing a circuit appeared to
        do nothing until the whole analysis was run again. Runs in a
        worker because a six-parameter circuit with a transmission line
        takes a few seconds, and a frozen window during those seconds is
        indistinguishable from a crash.
        """
        if not self.session.set_circuit(self.circuit_text_var.get()):
            self.flush_messages(self.session.messages)
            return
        self.circuit_text_var.set(self.session.circuit_text())
        self._show_circuit_note()
        self._fill_circuit_setup()
        self.run_async(self.session.refit_circuit, self._after_refit,
                       f"Ajustando {self.session.circuit_text()}…")

    def _after_refit(self, ok: bool) -> None:
        self.flush_messages(self.session.messages)
        if not ok:
            return
        self._fill_tables()
        self._draw_circuit()
        self._redraw()
        eis = getattr(self.session.result, "eis", None)
        fit = eis.fit if eis is not None else None
        if fit is not None:
            self.set_status(
                f"{self.session.circuit_text()}: χ² = {fit.chi_squared:.4g}, "
                f"residuo {fit.residual_pct:.3f} %")

    def _draw_circuit(self) -> None:
        """Show the circuit as a schematic rather than as a string."""
        set_text(self.circuit_drawing, self.session.circuit_drawing())

    def _compare_circuits(self) -> None:
        """Fit a ladder of circuits and show what separates them."""
        if self.session.eis is None:
            self.warn("Sin espectro",
                      "Carga primero un espectro de impedancia.")
            return
        self.run_async(self.session.compare_circuits, self._after_compare,
                       "Ajustando los circuitos candidatos…")

    def _after_compare(self, text: str) -> None:
        self.flush_messages(self.session.messages)
        self.show_text("Comparación de circuitos", text)

    def _export_fit_curve(self) -> None:
        from tkinter import filedialog

        if not self.session.fit_curve_rows():
            self.warn("Sin ajuste",
                      "Ajusta primero un circuito: la curva que se exporta "
                      "es la suya.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar la curva ajustada",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
            parent=self.root,
        )
        if not path:
            return
        self.session.save_fit_curve(path)
        self.flush_messages(self.session.messages)

    def _effective_capacitance(self) -> None:
        """The CPE's capacitance, by both relations, with the arithmetic."""
        text = self.session.effective_capacitance_report()
        if not text:
            self.warn(
                "Sin CPE ajustado",
                "Hace falta un circuito ajustado que lleve un CPE (Q) o un "
                "ZARC (Za). Un ZARC no necesita conversión: su capacitancia "
                "es τ/R.")
            return
        self.show_text("Capacitancia efectiva", text)

    def _save_circuit(self) -> None:
        from tkinter import simpledialog

        name = simpledialog.askstring(
            "Guardar circuito", "Nombre para este circuito:", parent=self.root)
        if not name:
            return
        if self.session.save_circuit(name, self.circuit_text_var.get()):
            self.circuit_box.configure(values=self.session.circuit_choices())
            self.circuit_var.set(name)
        self.flush_messages(self.session.messages)

    def _delete_circuit(self) -> None:
        name = self.circuit_var.get()
        if not self.ask_yes_no(
            "Borrar circuito",
            f"¿Borrar «{name}» de tus circuitos guardados? Los del "
            "programa no se borran.",
        ):
            return
        if self.session.delete_circuit(name):
            self.circuit_box.configure(values=self.session.circuit_choices())
            self.circuit_var.set(self.session.circuit)
        self.flush_messages(self.session.messages)

    def _fill_circuit_setup(self) -> None:
        fill_table(
            self.circuit_setup_table,
            ["parámetro", "valor de partida", "fijo"],
            [(label, f"{value:.6g}", "sí" if fixed else "")
             for label, value, fixed in self.session.circuit_parameters()],
        )

    def _selected_circuit_parameter(self):
        selection = self.circuit_setup_table.selection()
        if not selection:
            self.warn("Sin selección",
                      "Elige primero un parámetro de la tabla.")
            return None
        return self.circuit_setup_table.item(selection[0], "values")[0]

    def _edit_circuit_parameter(self, _event=None) -> None:
        from tkinter import simpledialog

        label = self._selected_circuit_parameter()
        if label is None:
            return
        current = next((v for lab, v, _ in self.session.circuit_parameters()
                        if lab == label), 0.0)
        value = simpledialog.askfloat(
            "Valor de partida", f"Valor de partida de {label}:",
            initialvalue=current, parent=self.root)
        if value is None:
            return
        self.session.set_circuit_parameter(label, value=value)
        self._fill_circuit_setup()

    def _toggle_circuit_fixed(self) -> None:
        label = self._selected_circuit_parameter()
        if label is None:
            return
        held = label in self.session.circuit_fixed
        self.session.set_circuit_parameter(label, fixed=not held)
        self._fill_circuit_setup()

    def _reset_circuit_parameters(self) -> None:
        self.session.reset_circuit_parameters()
        self._fill_circuit_setup()
        self.set_status("valores de partida leídos del espectro, nada fijado")

    def _settings_from_widgets(self) -> None:
        self.session.name = self.name_var.get() or "muestra"
        self.session.electrode = self._electrode_from_widgets()
        self.session.potential_scale = dict(POTENTIAL_UNITS).get(
            self.potential_unit_var.get(), 1.0)
        self.session.current_scale = dict(CURRENT_UNITS).get(
            self.current_unit_var.get(), 1.0)
        self.session.reaction = self.reaction_var.get()
        # NOT from the combobox: the box holds a name, and what the user
        # edited is the string in the entry. Taking the name back would
        # throw away every hand-made change the moment they pressed
        # Analizar, which is the one moment it has to survive.
        if hasattr(self, "circuit_text_var"):
            self.session.set_circuit(self.circuit_text_var.get())
        else:
            self.session.set_circuit(self.circuit_var.get())
        self.session.non_faradaic = bool(self.non_faradaic_var.get())
        self.session.cycle_choice = self._cycle_from_widget()
        labels = {label: key for key, label in POTENTIAL_SCALES}
        self.session.potential_scale_choice = labels.get(
            self.scale_var.get(), "medido")
        self.session.dunn_nonnegative = bool(self.dunn_nonneg_var.get())
        self.session.dunn_points = int(
            _number(self.dunn_points_var.get()) or 60)
        self.session.dunn_window = self._window_from(
            self.dunn_low_var, self.dunn_high_var)
        self.session.capacitance_window = self._window_from(
            self.cap_low_var, self.cap_high_var)
        self.session.subtract_parasitic = bool(self.subtract_var.get())
        self.session.retag_electrode()

    def _area_from_disc(self) -> None:
        diameter = _number(self.diameter_var.get())
        if diameter and diameter > 0:
            self.area_var.set(f"{Electrode().use_disc(diameter):.5g}")

    def _ph_from_electrolyte(self) -> None:
        """Fill the pH from the table when the name matches an entry."""
        chosen = self.electrolyte_var.get().strip()
        for key, entry in Electrode.electrolytes().items():
            if chosen in (key, entry["label"]):
                self.ph_var.set(f"{entry['ph']:g}")
                self.set_status(
                    f"{entry['label']}: pH {entry['ph']:g} "
                    f"(confianza {entry['confidence']}) — nominal, no medido"
                )
                return

    def _window_from(self, low_var, high_var) -> Optional[tuple[float, float]]:
        low, high = _number(low_var.get()), _number(high_var.get())
        if low is None or high is None or high <= low:
            return None
        return (low, high)

    def _export_tables(self) -> None:
        from tkinter import filedialog

        if not self.session.has_data:
            self.warn("No hay nada que exportar", "Carga una medida primero.")
            return
        folder = filedialog.askdirectory(
            title="Carpeta para las tablas", parent=self.root)
        if not folder:
            return
        self._settings_from_widgets()
        written = self.session.save_tables(folder)
        self.flush_messages(self.session.messages)
        if written:
            self.set_status(f"Escritos {len(written)} archivos en {folder}")

    def _cycle_from_widget(self) -> Optional[int]:
        label = self.cycle_var.get()
        for number, text in self.session.cycle_choices():
            if text == label:
                return number
        return None

    def _refresh_cycle_choices(self) -> None:
        """Rebuild the selector after loading, keeping the choice if it
        still exists. A selector that silently resets to the default
        every time a file is added undoes the one thing it is for."""
        choices = self.session.cycle_choices()
        labels = [text for _, text in choices]
        self.cycle_box.configure(values=labels)
        if self.cycle_var.get() not in labels:
            self.cycle_var.set(labels[0])
        self.session.cycle_choice = self._cycle_from_widget()
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
        elif kind == "gcd_series":
            ok = self.session.add_gcd_to_series(
                path, current=(current / 1000.0) if current else None
            )
        elif kind == "eis":
            ok = self.session.load_eis(path)
        elif kind == "eis_series":
            # The legend is asked for rather than taken from the file
            # name: what distinguishes several spectra is the sample, the
            # temperature or the potential, and the file name is almost
            # never that.
            from tkinter import simpledialog

            label = simpledialog.askstring(
                "Nombre en la leyenda",
                "¿Cómo se llama este espectro en la figura?\n"
                "(la muestra, la temperatura, el potencial…)",
                initialvalue=Path(path).stem, parent=self.root)
            ok = self.session.add_eis_to_series(path, label=label or "")
        elif kind == "lsv":
            ok = self.session.load_lsv(path)
        self.flush_messages(self.session.messages)
        if ok:
            self._refresh_cycle_choices()
            self._refresh_figure_panels()
            # Say what normalisation the numbers will use. A blank mass is
            # invisible until a capacitance comes back without its F/g,
            # three tabs later.
            electrode = self.session.electrode
            basis = (f"{electrode.mass_mg:g} mg" if electrode.mass_mg
                     else "SIN masa activa")
            basis += (f", {electrode.area_cm2:g} cm²" if electrode.area_cm2
                      else ", sin área")
            note = self.session.reporting_cycle_note()
            self.set_status(
                f"Cargado {Path(path).name} — {basis}"
                + (f" — {note.split('.')[0]}" if note else "")
            )
        self._refresh_loaded()

    def _load_eis_pair(self) -> None:
        """Two EC-Lab text exports that are one spectrum between them.

        Asked for in two steps rather than one multi-selection, because
        which file is which matters and a multi-selection returns them in
        whatever order the dialog felt like.
        """
        from tkinter import filedialog

        nyquist = filedialog.askopenfilename(
            title="Exportación Nyquist: Re(Z) y -Im(Z)",
            filetypes=FILE_TYPES, parent=self.root)
        if not nyquist:
            return
        bode = filedialog.askopenfilename(
            title="Exportación Bode de la MISMA medida: frecuencia y fase",
            filetypes=FILE_TYPES, parent=self.root)
        if not bode:
            return
        self._settings_from_widgets()
        if self.session.load_eis_pair(nyquist, bode):
            self._redraw()
        self.flush_messages(self.session.messages)

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
        self.electrolyte_var.set("KOH 6 M")
        self.ph_var.set("14")
        self.resistance_var.set("2.0")
        self.name_var.set("demo pseudocondensador híbrido")
        self.circuit_var.set("R0-(R1|Q1)-Q2")
        self._settings_from_widgets()
        self.session.add_curves(
            cv=make_cv_demo("hibrido", seed=1),
            rate_series=cv_rate_series("hibrido", seed=2),
            gcd=make_gcd_demo("hibrido", seed=1),
            # Several currents, because a Ragone plot with one point is
            # not a Ragone plot: the whole content of the figure is how
            # the energy falls as the power rises.
            gcd_series=[
                make_gcd_demo("hibrido", current=current, seed=seed)
                # 1 mA is left out: that is what `gcd` above already is,
                # and the same measurement twice on a Ragone plot reads
                # as two devices that happen to agree.
                for seed, current in enumerate((0.5e-3, 2e-3, 5e-3, 10e-3), start=10)
            ],
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
        fill_table(self.circuit_table,
                   ["parámetro", "valor", "incertidumbre", "fijo"],
                   self.session.circuit_rows())
        self._fill_circuit_setup()
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
            placeholder(ax, "Carga un voltamperograma", self.figure_palette)
            return
        result = self.session.result
        drawn, label, why = [], "", ""
        for curve in curves:
            values, label, why = self.session.potential_axis(curve)
            if np.allclose(values, curve.potential):
                drawn.append(curve)
                continue
            shifted = replace(curve, potential=values)
            drawn.append(shifted)
        plot_cv(
            ax, drawn, self.figure_palette,
            peaks=result.cv.peaks if result and result.cv else None,
            normalise_by_rate=bool(self.normalise_var.get()),
        )
        if label:
            ax.set_xlabel(label)
        if why and "no se ha podido" in why:
            self.session.log("warning", why)

    def _draw_rates(self, figure) -> None:
        from .plots_echem import plot_b_values, plot_dunn, plot_rate_capacitance

        left = figure.add_subplot(131)
        middle = figure.add_subplot(132)
        right = figure.add_subplot(133)
        result = self.session.result
        if result is None or result.rates is None:
            placeholder(left, "Carga una serie de velocidades", self.figure_palette)
            placeholder(middle, "", self.figure_palette)
            placeholder(right, "", self.figure_palette)
            return
        plot_rate_capacitance(left, result.rates, self.figure_palette)
        plot_b_values(middle, result.rates, self.figure_palette)
        if result.dunn is None:
            placeholder(right, "Dunn necesita tres velocidades", self.figure_palette)
        else:
            rate = self.session.dunn_rate_for_plot()
            curve = next(
                (c for c in self.session.rate_series
                 if rate is not None and abs(c.scan_rate - rate) < 1e-9),
                self.session.cv,
            )
            plot_dunn(right, result.dunn, curve, self.figure_palette,
                      scan_rate=rate, per_gram=True)

    def _draw_gcd(self, figure) -> None:
        from .plots_echem import plot_gcd

        ax = figure.add_subplot(111)
        if self.session.gcd is None:
            placeholder(ax, "Carga una curva de carga-descarga", self.figure_palette)
            return
        result = self.session.result
        plot_gcd(ax, self.session.gcd, self.figure_palette,
                 result.gcd if result else None)

    def _draw_rate_capacitance(self, figure) -> None:
        from .plots_echem import plot_capacitance_vs_current

        ax = figure.add_subplot(111)
        plot_capacitance_vs_current(ax, self.session.capacitance_vs_current(),
                                    self.figure_palette)

    def _draw_nyquist(self, figure) -> None:
        from .plots_echem import plot_nyquist

        ax = figure.add_subplot(111)
        if self.session.eis is None:
            placeholder(ax, "Carga un espectro de impedancia", self.figure_palette)
            return
        result = self.session.result
        plot_nyquist(ax, self.session.eis, self.figure_palette,
                     result.eis if result else None)

    def _draw_bode(self, figure) -> None:
        from .plots_echem import plot_bode, plot_kk_residuals

        left = figure.add_subplot(121)
        right = figure.add_subplot(122)
        if self.session.eis is None:
            placeholder(left, "", self.figure_palette)
            placeholder(right, "", self.figure_palette)
            return
        result = self.session.result
        plot_bode(left, self.session.eis, self.figure_palette, result.eis if result else None)
        if result is not None and result.eis is not None:
            plot_kk_residuals(right, result.eis, self.figure_palette)
        else:
            placeholder(right, "Analiza para ver Kramers-Kronig", self.figure_palette)

    def _draw_drt(self, figure) -> None:
        from .plots_echem import plot_drt

        ax = figure.add_subplot(111)
        result = self.session.result
        if result is None or result.drt is None:
            placeholder(ax, "Carga un espectro de impedancia y analiza",
                        self.figure_palette)
            return
        plot_drt(ax, result.drt, self.figure_palette)

    def _draw_capacitance(self, figure) -> None:
        from .plots_echem import plot_complex_capacitance

        left = figure.add_subplot(121)
        right = figure.add_subplot(122)
        result = self.session.result
        plot_complex_capacitance(
            left, right,
            result.complex_capacitance if result else None,
            self.figure_palette)

    def _draw_ragone(self, figure) -> None:
        from .plots_echem import plot_ragone

        ax = figure.add_subplot(111)
        points = self.session.ragone_points()
        if not points:
            placeholder(ax, "Carga una o varias curvas de carga-descarga",
                        self.figure_palette)
            return
        plot_ragone(ax, points, self.figure_palette)
        if len(points) == 1:
            ax.set_title(
                "un solo punto: añade curvas a otras corrientes",
                fontsize=9, color=self.figure_palette.text_muted)

    def _draw_polarisation(self, figure) -> None:
        from .plots_echem import plot_polarisation

        ax = figure.add_subplot(111)
        result = self.session.result
        plot_polarisation(ax, result.catalysis if result else None,
                          self.figure_palette)

    def _draw_tafel(self, figure) -> None:
        from .plots_echem import plot_tafel

        ax = figure.add_subplot(111)
        result = self.session.result
        curve = self.session.lsv
        if result is None or result.catalysis is None or curve is None:
            placeholder(ax, "Carga una curva de polarización", self.figure_palette)
            return
        catalysis = result.catalysis
        rhe, _ = curve.electrode.to_rhe(curve.potential)
        if rhe is None:
            placeholder(ax, "Hace falta el pH para pasar a RHE", self.figure_palette)
            return
        corrected, _ = curve.electrode.ir_correct(rhe, curve.current)
        equilibrium = 1.23 if catalysis.reaction == "OER" else 0.0
        sign = 1.0 if catalysis.reaction == "OER" else -1.0
        area = curve.electrode.area_cm2 or 1.0
        plot_tafel(
            ax, catalysis, self.figure_palette,
            overpotential=sign * (corrected - equilibrium),
            current_density=1e3 * curve.current / area,
        )


def _number(text: str) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _electrolyte_labels() -> list[str]:
    """Names for the electrolyte box, in the table's own order.

    Not sorted: the table is written with the common ones first, and an
    alphabetical list puts H2SO4 above KOH for no reason anybody cares
    about.
    """
    return [entry["label"] for entry in Electrode.electrolytes().values()]


def _reference_keys() -> list[str]:
    from ..echem.curve import load_echem_database

    return sorted(load_echem_database()["reference_electrodes"])


__all__ = ["FILE_TYPES", "EchemApp"]
