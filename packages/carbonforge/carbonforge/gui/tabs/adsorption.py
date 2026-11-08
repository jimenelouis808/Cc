"""The "Calcular → Adsorción (H₂)" page: porosity now, the isotherm in the queue.

The two halves of this page are split the way the module they drive is,
and the order is the whole point (see
:mod:`carbonforge.adsorption`): **geometry first, sampling second.**

* **Porosity runs here, on the spot.** It is seconds of Monte Carlo
  insertion, it needs no reservoir and no temperature, and it answers
  the question that decides whether the rest is worth doing — a
  framework whose largest free sphere is under the probe cannot adsorb
  whatever its cavities measure.
* **The isotherm goes to the job queue.** One state point on a
  validation-sized bundle runs 31 to 46 minutes, so an eight-point
  isotherm is most of a day. It is written as a
  ``carbonforge-isoterma/1`` folder and handed to the same queue that
  drives Quantum ESPRESSO, SIESTA and LAMMPS, which also means a
  cancelled run costs the point in flight and nothing before it.

So this page has no progress bar of its own and no cancel button: the
long half is a job, and jobs are listed, logged and cancelled on the
Trabajos page, where every other long calculation already is.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

from ase import Atoms

#: Temperatures with a published bulk reservoir. Offered as a menu rather
#: than a free field because the virial is fitted to measured densities
#: and these are the ones that exist; a typed 200 K would be refused
#: after the dialog had closed.
TEMPERATURES: tuple[str, ...] = ("77", "150", "293")

#: Categorical slots 1-8 of the validated palette, in fixed order. A
#: ninth isotherm is not a generated colour: the page refuses to load
#: it and says to remove one, because a repeated hue on a comparison
#: plot is worse than not drawing it.
SERIES_COLOURS: tuple[str, ...] = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#6250d6", "#e34948",
)

#: Levesque's own pressure ladder, which is also a sensible default
#: ladder for anything else: three decades with enough points at the
#: knee to see it.
DEFAULT_PRESSURES: str = "0.1, 0.5, 1, 3, 5, 10, 15, 20"


class AdsorptionTab:
    """Geometric porosity, and GCMC isotherms as queued jobs.

    Uses from the host: ``tk``, ``ttk``, ``root``, ``session``,
    ``_show_error``, ``submit_job`` (the queue) and ``select_page``.
    """

    # ------------------------------------------------------------------
    # Adsorption tab
    # ------------------------------------------------------------------
    def _build_adsorption_tab(self, parent) -> None:
        tk, ttk = self.tk, self.ttk

        outer = ttk.Frame(parent, padding=8)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer, width=360)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)
        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(
            left,
            text=("Adsorción de H₂ en un armazón rígido, por Monte Carlo "
                  "gran canónico."),
            wraplength=330, justify="left", foreground="#444444",
        ).pack(fill="x")
        ttk.Label(
            left,
            text=("El orden importa y por eso está numerado: la geometría "
                  "decide en segundos si el muestreo vale la pena, y el "
                  "muestreo cuesta horas. Si no tienes una estructura "
                  "cargada, hazla en «Construir» o tráela de un fichero en "
                  "«Importar» — cualquier .xyz, .cif o POSCAR, venga de "
                  "donde venga."),
            wraplength=330, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(fill="x", pady=(2, 0))
        ttk.Label(
            left,
            text=("Contrastado: este código reproduce las isotermas de "
                  "Levesque 2002 en 16 puntos a 77 y 293 K dentro de ±6 %. "
                  "Eso valida el muestreo, no el modelo — la molécula es un "
                  "solo sitio sin cuadrupolo, así que un armazón dopado "
                  "todavía no se describe bien. Ver docs/ADSORCION_MODELO.md."),
            wraplength=330, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(fill="x", pady=(4, 8))

        self.ads_structure_var = tk.StringVar(
            value="Nada todavía: construye o importa una estructura.")
        ttk.Label(left, textvariable=self.ads_structure_var, wraplength=330,
                  justify="left", foreground="#444444").pack(fill="x",
                                                             pady=(0, 8))

        # --- porosity ---------------------------------------------------
        poro = ttk.LabelFrame(left, text="1 · Porosidad geométrica", padding=6)
        poro.pack(fill="x")
        ttk.Label(
            poro,
            text=("Segundos, sin simulación. Dice si la molécula entra y si "
                  "puede cruzar la celda. Un armazón que falla aquí no "
                  "adsorbe nada, y no hace falta gastar horas para saberlo."),
            wraplength=330, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w")

        row = ttk.Frame(poro)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="Radio de sonda (Å)").pack(side="left")
        self.ads_probe_var = tk.StringVar(value="1.45")
        ttk.Entry(row, textvariable=self.ads_probe_var, width=6).pack(
            side="left", padx=(6, 0))
        ttk.Label(row, text="Muestras").pack(side="left", padx=(10, 0))
        self.ads_samples_var = tk.StringVar(value="200000")
        ttk.Entry(row, textvariable=self.ads_samples_var, width=9).pack(
            side="left", padx=(6, 0))
        ttk.Label(
            poro,
            text=("1.45 Å es medio diámetro cinético de H₂. El radio del "
                  "átomo del armazón es medio σ de H₂-C, así que «entra» "
                  "significa aquí lo mismo que para el GCMC — no es el "
                  "radio de van der Waals que usa parte de la literatura "
                  "de Zeo++."),
            wraplength=330, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w", pady=(4, 0))
        self.ads_poro_button = ttk.Button(
            poro, text="Medir la porosidad", state="disabled",
            command=self._on_measure_porosity)
        self.ads_poro_button.pack(fill="x", pady=(6, 0))

        # --- isotherm ---------------------------------------------------
        iso = ttk.LabelFrame(left, text="2 · Isoterma GCMC (a la cola)",
                             padding=6)
        iso.pack(fill="x", pady=(8, 0))
        ttk.Label(
            iso,
            text=("De 30 a 45 minutos POR PUNTO en un haz de tamaño de "
                  "validación, así que esto no se corre aquí: se escribe "
                  "una carpeta y se encola. Cada punto se guarda en cuanto "
                  "termina, y volver a lanzarla reanuda."),
            wraplength=330, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w")

        trow = ttk.Frame(iso)
        trow.pack(fill="x", pady=(4, 0))
        ttk.Label(trow, text="Temperatura (K)").pack(side="left")
        self.ads_temperature_var = tk.StringVar(value="77")
        ttk.Combobox(trow, textvariable=self.ads_temperature_var,
                     values=TEMPERATURES, state="readonly", width=6).pack(
            side="left", padx=(6, 0))
        ttk.Label(trow, text="Semilla").pack(side="left", padx=(10, 0))
        self.ads_seed_var = tk.StringVar(value="0")
        ttk.Entry(trow, textvariable=self.ads_seed_var, width=5).pack(
            side="left", padx=(6, 0))
        ttk.Label(
            iso,
            text=("Sólo 77, 150 y 293 K: el reservorio es un virial "
                  "ajustado a densidades publicadas, y son las que hay."),
            wraplength=330, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w", pady=(2, 0))

        ttk.Label(iso, text="Presiones (MPa)").pack(anchor="w", pady=(4, 0))
        self.ads_pressures_var = tk.StringVar(value=DEFAULT_PRESSURES)
        ttk.Entry(iso, textvariable=self.ads_pressures_var).pack(fill="x")

        mrow = ttk.Frame(iso)
        mrow.pack(fill="x", pady=(4, 0))
        ttk.Label(mrow, text="Equilibrado").pack(side="left")
        self.ads_equil_var = tk.StringVar(value="150000")
        ttk.Entry(mrow, textvariable=self.ads_equil_var, width=9).pack(
            side="left", padx=(6, 0))
        ttk.Label(mrow, text="Producción").pack(side="left", padx=(10, 0))
        self.ads_prod_var = tk.StringVar(value="300000")
        ttk.Entry(mrow, textvariable=self.ads_prod_var, width=9).pack(
            side="left", padx=(6, 0))
        ttk.Label(
            iso,
            text=("El equilibrado se descarta y sólo se promedia la "
                  "producción. Con estos valores la tabla de Levesque a "
                  "293 K sale dentro de ±6 %; si un punto avisa de que la "
                  "carga seguía derivando, súbelos y repítelo."),
            wraplength=330, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w", pady=(4, 0))

        srow = ttk.Frame(iso)
        srow.pack(fill="x", pady=(4, 0))
        ttk.Label(srow, text="Instantáneas por punto").pack(side="left")
        self.ads_snapshots_var = tk.StringVar(value="0")
        # ttk.Entry and not ttk.Spinbox: nothing else in this window uses
        # a Spinbox, and the GUI tests' ttk stub does not carry one -- the
        # first one added here broke 43 tests that have nothing to do with
        # adsorption, because a widget that raises in a build method takes
        # the whole page down with it.
        ttk.Entry(srow, textvariable=self.ads_snapshots_var,
                  width=5).pack(side="left", padx=(6, 0))
        ttk.Label(
            iso,
            text=("Configuraciones guardadas durante la producción, que es "
                  "de lo que se hacen las figuras de hidrógeno adsorbido. "
                  "Nunca del equilibrado: una celda que todavía se está "
                  "llenando parece un resultado y no lo es. Cuestan disco "
                  "y nada más; el muestreo no cambia por mirarlo."),
            wraplength=330, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w", pady=(4, 0))

        self.ads_queue_button = ttk.Button(
            iso, text="Escribir la carpeta y encolar…", state="disabled",
            command=self._on_queue_isotherm)
        self.ads_queue_button.pack(fill="x", pady=(6, 0))

        # --- reading them back, and comparing ---------------------------
        read = ttk.LabelFrame(left, text="3 · Leer y comparar", padding=6)
        read.pack(fill="x", pady=(8, 0))
        ttk.Button(read, text="Añadir una carpeta de isoterma…",
                   command=self._on_open_isotherm).pack(fill="x", ipady=3)
        ttk.Label(
            read,
            text=("Terminada o a medias: cada punto se guarda en cuanto "
                  "acaba, así que hay algo que mirar mucho antes del final. "
                  "Se AÑADEN, para poder comparar varias; «Abrir "
                  "resultados» en Trabajos trae aquí la seleccionada."),
            wraplength=330, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w", pady=(4, 0))

        self.ads_loaded: dict[str, dict] = {}
        # A one-column Treeview and not a Listbox: Treeview is what the
        # Trabajos page already uses for a list, and a Listbox is the
        # second widget this page tried to introduce alone -- the first
        # took 43 unrelated tests down with it, because a widget that
        # raises inside a build method kills the whole window.
        self.ads_list = ttk.Treeview(read, columns=(), show="tree",
                                     height=5, selectmode="browse")
        self.ads_list.pack(fill="x", pady=(4, 0))
        row = ttk.Frame(read)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Quitar", command=self._on_drop_isotherm).pack(
            side="left", expand=True, fill="x")
        ttk.Button(row, text="Vaciar",
                   command=self._on_clear_isotherms).pack(
            side="left", expand=True, fill="x", padx=(4, 0))
        ttk.Button(read, text="Exportar todo a CSV…",
                   command=self._on_export_isotherms).pack(fill="x",
                                                           pady=(4, 0))
        from ...adsorption.snapshot import ADSORBATE_COLOURS  # noqa: WPS433

        srow2 = ttk.Frame(read)
        srow2.pack(fill="x", pady=(4, 0))
        ttk.Label(srow2, text="Adsorbato").pack(side="left")
        self.ads_species_var = tk.StringVar(value="H2")
        ttk.Combobox(srow2, textvariable=self.ads_species_var, width=6,
                     values=tuple(ADSORBATE_COLOURS)).pack(side="left",
                                                           padx=(6, 0))
        ttk.Label(srow2, text="color").pack(side="left", padx=(10, 0))
        self.ads_colour_var = tk.StringVar(value="")
        ttk.Entry(srow2, textvariable=self.ads_colour_var,
                  width=9).pack(side="left", padx=(6, 0))
        self.ads_bonds_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(read, text="Dibujar los enlaces del armazón",
                        variable=self.ads_bonds_var).pack(anchor="w",
                                                          pady=(4, 0))
        ttk.Button(read, text="Ver el adsorbato…",
                   command=self._on_show_snapshot).pack(fill="x",
                                                        pady=(4, 0))
        ttk.Label(
            read,
            text=("El CSV sale en formato largo, con una columna «serie», "
                  "que es lo que cualquier programa de gráficas prefiere. "
                  "Las configuraciones sólo existen si se pidieron al "
                  "escribir la carpeta: cada sitio dibujado es UNA "
                  "MOLÉCULA, no un átomo — este modelo tiene un solo "
                  "centro por H₂. El color sale del catálogo de la "
                  "especie (rosa para H₂) y el campo de al lado lo "
                  "sobreescribe con cualquiera que entienda matplotlib: "
                  "un nombre, «#rrggbb», lo que sea."),
            wraplength=330, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w", pady=(4, 0))

        self.ads_status_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.ads_status_var, wraplength=330,
                  justify="left", foreground="#667").pack(anchor="w",
                                                          pady=(8, 0))

        # --- report and plot --------------------------------------------
        from matplotlib.backends.backend_tkagg import (  # noqa: WPS433
            FigureCanvasTkAgg,
        )
        from matplotlib.figure import Figure  # noqa: WPS433

        plot = ttk.LabelFrame(right, text="Isoterma", padding=6)
        plot.pack(fill="x")
        self.ads_figure = Figure(figsize=(5.2, 2.9), dpi=100,
                                 facecolor="#fcfcfb")
        self.ads_axis = self.ads_figure.add_subplot(111)
        self.ads_canvas = FigureCanvasTkAgg(self.ads_figure, master=plot)
        self.ads_canvas.get_tk_widget().pack(fill="both", expand=True)

        report = ttk.LabelFrame(right, text="Informe", padding=6)
        report.pack(fill="both", expand=True, pady=(8, 0))
        self.ads_text = tk.Text(report, wrap="word", font=("TkFixedFont", 9))
        bar = ttk.Scrollbar(report, orient="vertical",
                            command=self.ads_text.yview)
        self.ads_text.configure(yscrollcommand=bar.set, state="disabled")
        bar.pack(side="right", fill="y")
        self.ads_text.pack(side="left", fill="both", expand=True)

        # Only now: `_report_isotherms` writes into `ads_text` and
        # `_plot_isotherms` draws on `ads_axis`, so the empty state has
        # to be painted after BOTH exist. Calling it beside the canvas
        # raised AttributeError on the text widget, which in a Tk build
        # method means a half-built page and no traceback anyone sees.
        self._plot_isotherms()
        self._report_isotherms()

        session = getattr(self, "session", None)
        if session is not None:
            session.subscribe(self._on_adsorption_structure)

    # -- state ---------------------------------------------------------

    def _adsorption_atoms(self) -> Optional[Atoms]:
        """The window's current structure, as every other page takes it."""
        session = getattr(self, "session", None)
        if session is not None:
            return session.take()
        return getattr(self, "atoms", None)

    def _on_adsorption_structure(self, current) -> None:
        state = "disabled" if current is None else "normal"
        self.ads_poro_button.configure(state=state)
        self.ads_queue_button.configure(state=state)
        self.ads_structure_var.set(
            "Nada todavía: construye o importa una estructura."
            if current is None else current.describe())

    def _set_adsorption_report(self, text: str) -> None:
        self.ads_text.configure(state="normal")
        self.ads_text.delete("1.0", "end")
        self.ads_text.insert("1.0", text)
        self.ads_text.configure(state="disabled")

    # -- porosity ------------------------------------------------------

    def _on_measure_porosity(self) -> None:
        """Measure the current structure's porosity and print the report.

        Run inline, which is a decision rather than a shortcut: at the
        default sample count this is a few seconds on a framework of a
        few thousand atoms, and routing it through the job queue would
        put a folder on disk and a row in a list for something the user
        is waiting on a sentence from.
        """
        from ...adsorption import analyse_porosity

        atoms = self._adsorption_atoms()
        if atoms is None:
            return
        try:
            probe = float(self.ads_probe_var.get())
            samples = int(self.ads_samples_var.get())
            if probe <= 0.0:
                raise ValueError("El radio de sonda tiene que ser positivo.")
            if samples < 1000:
                raise ValueError(
                    "Menos de 1000 muestras no mide nada: el error de la "
                    "fracción de vacío va como 1/sqrt(muestras).")
        except ValueError as exc:
            self._show_error(exc, traceback.format_exc())
            return

        self.ads_status_var.set("Midiendo la porosidad…")
        self.root.update_idletasks()
        try:
            report = analyse_porosity(atoms, probe_radius=probe,
                                      samples=samples, seed=0)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.ads_status_var.set("La porosidad falló.")
            self._show_error(exc, traceback.format_exc())
            return

        crosses = report.largest_free_sphere >= 2.0 * report.probe_radius
        lines = [
            f"{atoms.get_chemical_formula()} · {len(atoms)} átomos",
            "",
            report.summary(),
            "",
            ("La sonda CRUZA la celda, así que el GCMC tiene sentido: "
             "sigue en el paso 2. Para un primer tanteo deja las ocho "
             "presiones y mira el punto más alto; para una isoterma "
             "publicable sube el equilibrado hasta que ningún punto avise "
             "de deriva."
             if crosses else
             "La sonda NO cruza la celda: este armazón está cerrado a esta "
             "molécula y adsorberá esencialmente nada, por grandes que "
             "sean sus cavidades. Un GCMC confirmaría esto en horas."),
            "",
            "El área es geométrica (una sonda rodada sobre esferas duras), "
            "no BET. BET es un modelo ajustado a una isoterma de nitrógeno "
            "medida; decenas por ciento de diferencia es normal entre las "
            "dos y tres cifras de acuerdo serían sospechosas.",
        ]
        self._set_adsorption_report("\n".join(lines))
        self.ads_status_var.set(
            "Porosidad medida." if crosses
            else "Porosidad medida: la sonda no cruza la celda.")

    # -- reading a finished isotherm back ------------------------------

    def _on_open_isotherm(self) -> None:
        """Pick a finished (or part-finished) isotherm folder and add it."""
        from tkinter import filedialog

        folder = filedialog.askdirectory(
            title="Carpeta de isoterma (la que tiene isoterma.json)")
        if folder:
            self._show_isotherm(Path(folder))

    def _on_drop_isotherm(self) -> None:
        """Remove the selected isotherm from the comparison."""
        name = self._selected_isotherm()
        if name is None:
            return
        del self.ads_loaded[name]
        self._refresh_isotherm_list()

    def _selected_isotherm(self) -> Optional[str]:
        """The highlighted series' name, or None."""
        chosen = self.ads_list.selection()
        if not chosen:
            return None
        name = self.ads_list.item(chosen[0], "text")
        return name if name in self.ads_loaded else None

    def _on_clear_isotherms(self) -> None:
        self.ads_loaded.clear()
        self._refresh_isotherm_list()

    def _refresh_isotherm_list(self) -> None:
        """Redraw the list, the plot and the report from what is loaded."""
        for item in self.ads_list.get_children():
            self.ads_list.delete(item)
        for name in self.ads_loaded:
            self.ads_list.insert("", "end", text=name)
        self._plot_isotherms()
        self._report_isotherms()

    def _on_export_isotherms(self) -> None:
        """Write every loaded isotherm as one tidy CSV.

        One file with a ``serie`` column rather than one file each,
        because the reason to load several is to plot them together and
        every plotting program would rather be handed long-form data
        than asked to join four files.
        """
        from tkinter import filedialog

        if not self.ads_loaded:
            self.ads_status_var.set("No hay ninguna isoterma cargada.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar las isotermas cargadas",
            defaultextension=".csv",
            initialfile="isotermas.csv",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")])
        if not path:
            return
        columns = ("serie", "temperatura_K", "presion_MPa", "wt_pct_total",
                   "wt_pct_por_carbono", "rho_absoluta_g_cm3",
                   "rho_exceso_g_cm3", "rho_gas_g_cm3", "N_medio", "N_error")
        lines = [
            "# rho_absoluta: todas las moleculas de la celda. "
            "rho_exceso: lo que mide un experimento.",
            "# wt_pct_total = 100 m(H2)/(m(H2)+m(C)), la definicion de "
            "los articulos.",
            ",".join(columns),
        ]
        for name, entry in self.ads_loaded.items():
            temperature = entry["request"]["temperatura_k"]
            for row in entry["points"]:
                lines.append(",".join([
                    name.replace(",", " "), f"{temperature:g}",
                    f"{row['presion_mpa']:.6g}",
                    f"{row['wt_pct_total']:.6g}",
                    f"{row['wt_pct_por_carbono']:.6g}",
                    f"{row['densidad_absoluta_g_cm3']:.6g}",
                    f"{row['densidad_exceso_g_cm3']:.6g}",
                    f"{row['densidad_gas_g_cm3']:.6g}",
                    f"{row['n_medio']:.6g}", f"{row['n_error']:.6g}"]))
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        total = sum(len(e["points"]) for e in self.ads_loaded.values())
        self.ads_status_var.set(
            f"{total} puntos de {len(self.ads_loaded)} isotermas en {path}")

    def _on_show_snapshot(self) -> None:
        """Open the selected isotherm's saved configuration, in 3 views."""
        from tkinter import filedialog

        from ...adsorption.snapshot import ADSORBATE_TAG, plot_snapshot

        name = self._selected_isotherm()
        if name is None:
            self.ads_status_var.set(
                "Elige una isoterma de la lista primero.")
            return
        entry = self.ads_loaded[name]
        folder = entry["folder"] / "instantaneas"
        if not folder.is_dir() or not any(folder.iterdir()):
            self.ads_status_var.set(
                "Esa isoterma no guardó configuraciones. Se piden al "
                "escribir la carpeta, en «Instantáneas por punto».")
            return
        path = filedialog.askopenfilename(
            title="Configuración", initialdir=str(folder),
            filetypes=[("extxyz", "*.extxyz"), ("Todos", "*.*")])
        if not path:
            return
        from ase.io import read

        atoms = read(path)
        tags = atoms.get_tags()
        species = self.ads_species_var.get().strip() or "H2"
        colour = self.ads_colour_var.get().strip() or None
        try:
            figure = plot_snapshot(
                atoms[tags != ADSORBATE_TAG],
                atoms[tags == ADSORBATE_TAG].get_positions(),
                view="todas", species=species, colour=colour,
                bonds=bool(self.ads_bonds_var.get()))
        except ValueError as exc:
            # An unknown matplotlib colour is a typo, not a crash.
            self.ads_status_var.set(f"Color no reconocido: {colour}")
            self._show_error(exc, traceback.format_exc())
            return
        window = self.tk.Toplevel(self.root)
        window.title(Path(path).name)
        from matplotlib.backends.backend_tkagg import (  # noqa: WPS433
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )

        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas, window).update()
        canvas.draw()
        self.ads_status_var.set(f"Configuración: {Path(path).name}")

    def _show_isotherm(self, folder: Path) -> None:
        """Add a folder to the comparison and redraw.

        Adds rather than replaces, because the reason to open a second
        one is almost always to put it beside the first. Reached from
        the folder picker here and from «Abrir resultados» on the
        Trabajos page. Until this existed a run of several hours
        finished and the window said it had produced nothing to show,
        which left reading a JSON by hand as the only way to see your
        own result.

        It takes a **part-finished** folder too, and on purpose: the
        format appends each point as it lands, so there is something
        worth looking at long before the job is done.
        """
        from ...adsorption.campaign import (read_isotherm_request,
                                            read_results)

        try:
            request = read_isotherm_request(folder)
        except (FileNotFoundError, ValueError) as exc:
            self.ads_status_var.set("Esa carpeta no es una isoterma.")
            self._show_error(exc, traceback.format_exc())
            return

        name = f"{request['nombre']} · {request['temperatura_k']:g} K"
        if name in self.ads_loaded and self.ads_loaded[name][
                "folder"] != folder:
            name = f"{name} · {folder.name}"
        if (len(self.ads_loaded) >= len(SERIES_COLOURS)
                and name not in self.ads_loaded):
            self.ads_status_var.set(
                f"Ya hay {len(SERIES_COLOURS)} isotermas cargadas, que son "
                "los colores que esta paleta distingue con seguridad. Quita "
                "una antes de añadir otra.")
            return
        self.ads_loaded[name] = {
            "folder": folder,
            "request": request,
            "points": read_results(folder),
        }
        self._refresh_isotherm_list()

    def _report_isotherms(self) -> None:
        """The table under the plot: every loaded isotherm, in order."""
        from ...adsorption.campaign import status

        if not self.ads_loaded:
            self._set_adsorption_report(
                "Ninguna isoterma cargada. Abre una carpeta en el paso 3, o "
                "usa «Abrir resultados» en Calcular → Trabajos.")
            return

        lines: list[str] = []
        for name, entry in self.ads_loaded.items():
            info = status(entry["folder"])
            request = entry["request"]
            lines += [
                name,
                f"  {entry['folder']}",
                f"  {request['atomos']} átomos · {info['line']} — "
                f"{info['state']}",
            ]
            points = entry["points"]
            if not points:
                lines += ["  Todavía ningún punto terminado. El primero "
                          "aparece en cuanto acaba, no al final.", ""]
                continue
            muestreo = request["muestreo"]
            lines += [
                f"  {'P (MPa)':>8} {'wt% total':>10} {'wt% /C':>8} "
                f"{'rho_o':>9} {'exceso':>10} {'N':>11}",
            ]
            for row in points:
                lines.append(
                    f"  {row['presion_mpa']:8.2f} {row['wt_pct_total']:10.2f} "
                    f"{row['wt_pct_por_carbono']:8.2f} "
                    f"{row['densidad_absoluta_g_cm3']:9.5f} "
                    f"{row['densidad_exceso_g_cm3']:+10.5f} "
                    f"{row['n_medio']:8.0f}±{row['n_error']:.0f}")
                for warning in row.get("avisos", []):
                    lines.append(f"      aviso: {warning}")
            lines += [
                f"  {muestreo['equilibrado']} de equilibrado y "
                f"{muestreo['produccion']} de producción por punto, "
                f"semilla {muestreo['semilla']}.",
                "",
            ]
        lines += [
            "wt% total es 100·m(H₂)/(m(H₂)+m(C)), que es la definición que "
            "usan los artículos y la que conviene citar. La de por carbono "
            "está al lado porque difieren un 5 % a 5 wt% y hay que decir "
            "cuál se usa.",
            "",
            "rho_o es ABSOLUTO: todas las moléculas de la celda. El exceso "
            "es lo que mide un experimento — lo que hay por encima de lo "
            "que ese mismo volumen tendría como gas. A 77 K y presión alta "
            "la diferencia es de un tercio, así que comparar un absoluto "
            "simulado con un exceso medido es el error más común de esta "
            "literatura.",
        ]
        self._set_adsorption_report("\n".join(lines))

    def _plot_isotherms(self) -> None:
        """Overlay every loaded isotherm, absolute and excess.

        Colour follows the series, assigned in load order from a fixed
        palette and never recycled: removing one must not repaint the
        others, or a reader who learned which curve was which is
        misled. Absolute is a solid line and excess a dashed one of the
        same colour, so the pair reads as one material rather than two.
        """
        axis = self.ads_axis
        axis.clear()
        axis.set_facecolor("#fcfcfb")
        axis.grid(True, color="#e1e0d9", lw=0.7)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.set_xlabel("presión (MPa)", fontsize=8)
        axis.set_ylabel("wt% (masa total)", fontsize=8)
        axis.tick_params(labelsize=7, colors="#898781", labelcolor="#52514e")

        drawn = False
        for index, (name, entry) in enumerate(self.ads_loaded.items()):
            points = entry["points"]
            if not points:
                continue
            drawn = True
            colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
            pressures = [row["presion_mpa"] for row in points]
            absolute = [row["wt_pct_total"] for row in points]
            excess = [
                row["wt_pct_total"] * row["densidad_exceso_g_cm3"]
                / row["densidad_absoluta_g_cm3"]
                if row["densidad_absoluta_g_cm3"] else 0.0
                for row in points]
            axis.plot(pressures, absolute, "-o", color=colour, lw=1.8,
                      ms=5.5, label=name)
            axis.plot(pressures, excess, "--", color=colour, lw=1.3,
                      alpha=0.8)
        if drawn:
            axis.legend(frameon=False, fontsize=7, labelcolor="#52514e")
            axis.annotate("continuo: absoluto · discontinuo: exceso",
                          xy=(0.99, 0.02), xycoords="axes fraction",
                          ha="right", va="bottom", fontsize=7,
                          color="#898781")
        self.ads_figure.tight_layout()
        self.ads_canvas.draw_idle()

    # -- isotherm ------------------------------------------------------

    def _parse_pressures(self) -> list[float]:
        """The pressure field, as a list. Raises ValueError with a sentence."""
        raw = self.ads_pressures_var.get().replace(",", " ").split()
        if not raw:
            raise ValueError(
                "No hay presiones: una isoterma necesita al menos una.")
        values = []
        for item in raw:
            try:
                values.append(float(item))
            except ValueError:
                raise ValueError(
                    f"«{item}» no es un número. Separa las presiones con "
                    "comas o espacios, en MPa.") from None
        return values

    def _on_queue_isotherm(self) -> None:
        """Write a request folder for the current structure and queue it."""
        from tkinter import filedialog

        from ...adsorption import write_isotherm_request

        atoms = self._adsorption_atoms()
        if atoms is None:
            return
        try:
            pressures = self._parse_pressures()
            temperature = float(self.ads_temperature_var.get())
            equilibration = int(self.ads_equil_var.get())
            production = int(self.ads_prod_var.get())
            seed = int(self.ads_seed_var.get())
            probe = float(self.ads_probe_var.get())
            samples = int(self.ads_samples_var.get())
            snapshots = int(self.ads_snapshots_var.get())
        except ValueError as exc:
            self._show_error(exc, traceback.format_exc())
            return

        folder = filedialog.askdirectory(
            title="Carpeta vacía para la isoterma")
        if not folder:
            return

        try:
            written = write_isotherm_request(
                atoms, folder, temperature=temperature, pressures=pressures,
                equilibration=equilibration, production=production, seed=seed,
                probe_radius=probe, porosity_samples=samples,
                snapshots=snapshots,
                reason="Encolada desde la ventana de carbonforge.")
        except ValueError as exc:
            self.ads_status_var.set("No se escribió la carpeta.")
            self._show_error(exc, traceback.format_exc())
            return

        hours = len(pressures) * 0.6
        lines = [
            f"Isoterma escrita en {written}",
            "",
            f"{atoms.get_chemical_formula()} · {len(atoms)} átomos",
            f"{temperature:g} K, {len(pressures)} presiones: "
            + ", ".join(f"{p:g}" for p in sorted(set(pressures))) + " MPa",
            f"{equilibration} movimientos de equilibrado y {production} de "
            f"producción por punto, semilla {seed}",
            "",
            f"Encolada. En un haz de tamaño de validación esto son del "
            f"orden de {hours:.0f} horas; en la página Trabajos se ve el "
            f"avance punto por punto, y se puede cancelar sin perder los "
            f"puntos que ya estén.",
            "",
            "Cada punto se guarda en resultados.json en cuanto termina. "
            "Volver a encolar la misma carpeta reanuda en vez de empezar: "
            "los puntos que ya están no se recalculan.",
            "",
            "Las dos convenciones de wt% se guardan con nombre "
            "(«wt_pct_total» y «wt_pct_por_carbono»). Difieren un 5 % a "
            "5 wt%, que es el tamaño entero de la corrección cuántica a "
            "77 K, y los artículos no dicen cuál usan.",
        ]
        self._set_adsorption_report("\n".join(lines))

        job = self.submit_job(Path(written), nprocs=1)
        if job is None:
            self.ads_status_var.set(
                f"Carpeta escrita en {written}, pero no se pudo encolar. "
                "Añádela a mano en Trabajos, o córrela con "
                f"python -m carbonforge.adsorption.run {written}")
            return
        self.ads_status_var.set(
            f"Encolada: {written}. El avance está en Calcular → Trabajos.")
        select = getattr(self, "select_page", None)
        if select is not None:
            select("Trabajos")
