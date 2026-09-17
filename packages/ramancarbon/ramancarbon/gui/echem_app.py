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
from ..echem.cv import SWEEP_CHOICES
from .base import SectionApp, placeholder
from .echem_state import EchemSession
from .theme import PAD
from .widgets import (
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
            "rate_capacitance": self._draw_rate_capacitance,
            "nyquist": self._draw_nyquist,
            "bode": self._draw_bode,
            "drt": self._draw_drt,
            "capacitance": self._draw_capacitance,
            "ragone": self._draw_ragone,
            "tafel": self._draw_tafel,
            "polarisation": self._draw_polarisation,
        }
        self._tab_canvases = {
            0: (),
            1: ("cv", "rates"),
            2: ("gcd", "rate_capacitance"),
            3: ("nyquist", "bode"),
            4: ("drt",),
            5: ("capacitance", "ragone"),
            6: ("tafel", "polarisation"),
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
        self.volume_var = tk.StringVar(value="")
        labelled(ebody, "Volumen (cm³)", lambda p: ttk.Entry(
            p, textvariable=self.volume_var, width=10))
        self.electrolyte_var = tk.StringVar(value="")
        labelled(ebody, "Electrolito", lambda p: ttk.Entry(
            p, textvariable=self.electrolyte_var, width=14))
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
            ("Añadir a la serie GCD…", lambda: self._load("gcd_series")),
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
        self.circuit_box = labelled(obody, "Circuito", lambda p: ttk.Combobox(
            p, textvariable=self.circuit_var, width=16,
            values=self.session.circuit_choices()))
        self.circuit_var.trace_add("write", lambda *_: self._on_circuit_chosen())
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

        dunn_bar = ttk.Frame(tab)
        dunn_bar.pack(fill="x", pady=(0, PAD["sm"]))
        ttk.Label(dunn_bar, text="Dunn — rama:").pack(side="left")
        self.dunn_sweep_var = self.tk.StringVar(value=SWEEP_CHOICES[0][1])
        ttk.Combobox(
            dunn_bar, textvariable=self.dunn_sweep_var, width=38,
            state="readonly",
            values=[label for _, label in SWEEP_CHOICES],
        ).pack(side="left", padx=(PAD["xs"], PAD["md"]))
        ttk.Label(dunn_bar, text="velocidad del gráfico (mV/s):").pack(side="left")
        self.dunn_rate_var = self.tk.StringVar(value="")
        self.dunn_rate_box = ttk.Combobox(
            dunn_bar, textvariable=self.dunn_rate_var, width=10,
            state="readonly", values=[])
        self.dunn_rate_box.pack(side="left", padx=(PAD["xs"], PAD["xs"]))
        self.dunn_rate_box.bind("<<ComboboxSelected>>",
                                lambda _e: self._on_dunn_rate())
        ttk.Button(dunn_bar, text="Recalcular Dunn",
                   command=self._recompute_dunn).pack(side="left")
        hint(dunn_bar,
             "  El gráfico se dibuja a la velocidad MÁS LENTA por defecto, "
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
        ttk.Entry(row, textvariable=self.circuit_text_var, width=44).pack(
            side="left", fill="x", expand=True, padx=(PAD["sm"], PAD["sm"]))
        ttk.Button(row, text="Aplicar", command=self._apply_circuit).pack(
            side="left")
        ttk.Button(row, text="Guardar como…",
                   command=self._save_circuit).pack(side="left",
                                                    padx=(PAD["xs"], 0))
        ttk.Button(row, text="Borrar el mío",
                   command=self._delete_circuit).pack(side="left",
                                                      padx=(PAD["xs"], 0))
        self.circuit_note = ttk.Label(
            editor_body, text="", wraplength=900, justify="left",
            style="Hint.TLabel")
        self.circuit_note.pack(fill="x", pady=(PAD["xs"], 0))
        hint(editor_body,
             "- serie, | paralelo, paréntesis agrupan. Elementos: R "
             "resistencia, C condensador, L inductancia, Q CPE, W Warburg "
             "semiinfinito, Ws finito transmisivo, Wo finito reflectante, "
             "G Gerischer, T electrodo poroso (línea de transmisión). "
             "Ejemplo: R0-(R1|Q1)-T1.",
             wrap=900)

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
        self.set_status(f"circuito: {self.session.circuit_text()}")

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
        plot_cv(
            ax, curves, self.figure_palette,
            peaks=result.cv.peaks if result and result.cv else None,
            normalise_by_rate=bool(self.normalise_var.get()),
        )

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
                      scan_rate=rate)

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


def _reference_keys() -> list[str]:
    from ..echem.curve import load_echem_database

    return sorted(load_echem_database()["reference_electrodes"])


__all__ = ["FILE_TYPES", "EchemApp"]
