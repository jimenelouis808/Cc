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

        self.ads_queue_button = ttk.Button(
            iso, text="Escribir la carpeta y encolar…", state="disabled",
            command=self._on_queue_isotherm)
        self.ads_queue_button.pack(fill="x", pady=(6, 0))

        self.ads_status_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.ads_status_var, wraplength=330,
                  justify="left", foreground="#667").pack(anchor="w",
                                                          pady=(8, 0))

        # --- report -----------------------------------------------------
        report = ttk.LabelFrame(right, text="Informe", padding=6)
        report.pack(fill="both", expand=True)
        self.ads_text = tk.Text(report, wrap="word", font=("TkFixedFont", 9))
        bar = ttk.Scrollbar(report, orient="vertical",
                            command=self.ads_text.yview)
        self.ads_text.configure(yscrollcommand=bar.set, state="disabled")
        bar.pack(side="right", fill="y")
        self.ads_text.pack(side="left", fill="both", expand=True)

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
            ("La sonda CRUZA la celda, así que el GCMC tiene sentido."
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
