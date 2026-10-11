"""The "Estructura → Construir" page: geometry, doping, groups; build and check.

Structures are built on a worker thread; the host's queue brings the result
back to the Tk thread (``_on_built``), which publishes it as the window's
current structure. The calculation is prepared on the Preparar page.
"""

from __future__ import annotations

import threading
import time
import traceback

from ase import Atoms

from ...validation.checks import run_basic_checks
from ..params import (
    FUNCTIONALIZATION_PARAMS,
    IMPORTED_KIND,
    MODIFIER_PARAMS,
    STRUCTURES,
    apply_functionalization,
    apply_modifiers,
    build_structure,
    check_parameter_constraints,
    describe_structure,
    validate_calculation,
)

def _clock(seconds: float) -> str:
    """Tiempo transcurrido como lo lee un cronómetro.

    Segundos por debajo del minuto y m:ss por encima, porque «143 s»
    obliga a dividir y «2:23» no.
    """
    if seconds < 60.0:
        return f"{seconds:.0f} s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}:{rest:02d}"


class BuilderTab:
    """Geometry on the left, preview on the right; build and check.

    The calculation (recipe, settings, export) is prepared on the Preparar
    page from the window's current structure, which this page publishes.

    Uses from the host: ``tk``, ``ttk``, ``root``, ``atoms``, the ``_*_vars``
    stores, ``_queue``, ``_add_field``, ``_read_raw``, ``_show_error``, and the
    preview panel (``_build_preview``, ``_render``, ``_discard_current_structure``).
    """

    def _build_builder_tab(self, parent) -> None:
        tk, ttk = self.tk, self.ttk

        # La estructura traida tal cual, intacta: cada construccion parte de
        # ella y no del resultado de la anterior.
        self._imported_base = None

        outer = ttk.Frame(parent, padding=8)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer, width=380)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        # --- structure selector -----------------------------------------
        sel = ttk.LabelFrame(left, text="Tipo de estructura", padding=8)
        sel.pack(fill="x")

        self._structure_labels = {v.label: k for k, v in STRUCTURES.items()}
        self.structure_var = tk.StringVar(
            value=STRUCTURES["cnt"].label
        )
        combo = ttk.Combobox(
            sel,
            textvariable=self.structure_var,
            values=list(self._structure_labels),
            state="readonly",
        )
        combo.pack(fill="x")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._rebuild_param_fields())

        self.description_label = ttk.Label(
            sel, text="", wraplength=340, justify="left", foreground="#444444"
        )
        self.description_label.pack(fill="x", pady=(6, 0))

        # --- scrollable parameter area ----------------------------------
        params_box = ttk.LabelFrame(left, text="Parámetros", padding=4)
        params_box.pack(fill="both", expand=True, pady=(8, 0))

        canvas = tk.Canvas(params_box, highlightthickness=0, width=340)
        scroll = ttk.Scrollbar(params_box, orient="vertical", command=canvas.yview)
        self.params_frame = ttk.Frame(canvas)
        self.params_frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.params_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # --- action buttons ---------------------------------------------
        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=(8, 0))

        self.build_button = ttk.Button(
            actions, text="Construir y previsualizar", command=self._on_build
        )
        self.build_button.pack(fill="x")

        ttk.Button(
            actions, text="Comprobar parámetros",
            command=self._on_check_constraints,
        ).pack(fill="x", pady=(4, 0))

        # A structure made on another page (a vibspec model, an import):
        # bring it here to see it, check it or keep decorating it.
        ttk.Button(
            actions, text="Traer la estructura actual",
            command=self._on_take_current,
        ).pack(fill="x", pady=(4, 0))

        # La geometria se podia mandar a Preparar, pero no sacarla a un
        # archivo desde aqui: para abrirla en otro programa habia que
        # exportar un calculo entero.
        self.save_button = ttk.Button(
            actions, text="Guardar estructura…", command=self._on_save_structure,
            state="disabled",
        )
        self.save_button.pack(fill="x", pady=(4, 0))

        self.png_button = ttk.Button(
            actions, text="Guardar imagen PNG…", command=self._on_save_png,
            state="disabled",
        )
        self.png_button.pack(fill="x", pady=(4, 0))

        # Problems whose cure is a setting, each with a button that applies it.
        self.fix_frame = ttk.LabelFrame(left, text="Correcciones", padding=4)
        self.fix_frame.pack(fill="x", pady=(8, 0))
        self._fix_frames.append(self.fix_frame)
        ttk.Label(self.fix_frame, text="Comprueba o construye para ver qué se puede corregir.",
                  foreground="#777777", wraplength=330).pack(anchor="w")

        self.status_var = tk.StringVar(value="Listo.")
        ttk.Label(
            left, textvariable=self.status_var, wraplength=340,
            justify="left", foreground="#0a6",
        ).pack(fill="x", pady=(8, 0))
        # Una construcción larga no daba señal alguna de estar viva.
        self.elapsed_var = tk.StringVar(value="")
        ttk.Label(
            left, textvariable=self.elapsed_var, justify="left",
            foreground="#667", font=("TkDefaultFont", 8),
        ).pack(fill="x")

        # --- right: preview + info --------------------------------------
        self._build_preview(right)

    # ------------------------------------------------------------------
    # Dynamic parameter fields
    # ------------------------------------------------------------------
    def _current_structure_key(self) -> str:
        return self._structure_labels[self.structure_var.get()]

    def _rebuild_param_fields(self) -> None:
        ttk = self.ttk
        for child in self.params_frame.winfo_children():
            child.destroy()
        self._param_vars.clear()
        self._modifier_vars.clear()
        self._functionalization_vars.clear()

        # Switching structure type invalidates whatever was built before;
        # otherwise "Exportar" would silently write the previous structure.
        self._discard_current_structure()

        spec = STRUCTURES[self._current_structure_key()]
        if spec.key != IMPORTED_KIND:
            # Elegir un preset es decir "genera esto": la base importada deja
            # de aplicar y guardarla solo serviria para resucitarla sola.
            self._imported_base = None
        self.description_label.configure(text=spec.description)

        geometry = ttk.LabelFrame(
            self.params_frame, text="1. Geometría", padding=4
        )
        geometry.pack(fill="x", pady=(10, 0))
        for param in spec.params:
            self._add_field(geometry, param, self._param_vars)

        if spec.supports_modifiers:
            mods = ttk.LabelFrame(
                self.params_frame, text="2. Dopaje y defectos", padding=4
            )
            mods.pack(fill="x", pady=(10, 0))
            for param in MODIFIER_PARAMS:
                self._add_field(mods, param, self._modifier_vars)

            groups = ttk.LabelFrame(
                self.params_frame,
                text="3. Grupos funcionales y nitrógeno",
                padding=4,
            )
            groups.pack(fill="x", pady=(10, 0))
            for param in FUNCTIONALIZATION_PARAMS:
                self._add_field(groups, param, self._functionalization_vars)

        ttk.Label(
            self.params_frame,
            text="El cálculo (receta, ajustes, formatos) se prepara en Preparar → "
                 "Cálculo, sobre la estructura actual.",
            wraplength=320, justify="left", foreground="#777777",
        ).pack(fill="x", pady=(10, 0))

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------
    def _on_check_constraints(self) -> None:
        """Report incompatible parameter combinations before building.

        Per-field bounds catch a bad number alone; this catches numbers that
        are each fine but wrong together, which is the commoner mistake.
        """
        values = self._all_values()
        # Several rules need the structure; build it if we can, but never let
        # a build failure hide the parameter feedback.
        atoms = self.atoms
        if atoms is None:
            try:
                atoms = build_structure(
                    self._current_structure_key(),
                    self._read_raw(self._param_vars),
                )
            except Exception:
                atoms = None
        self._set_info(check_parameter_constraints(atoms, values))
        self._refresh_fixes(atoms)
        self.status_var.set("Parámetros comprobados.")

    def _on_take_current(self) -> None:
        """Adopt the window's current structure (from another page)."""
        session = getattr(self, "session", None)
        current = session.current if session is not None else None
        if current is None:
            self.status_var.set("No hay estructura actual: construye, importa o arma un "
                                "modelo en otra página.")
            return
        if current is getattr(self, "_published", None):
            self.status_var.set("La estructura actual ya es la de esta página.")
            return
        atoms = session.take()
        # Cambiar de tipo descarta lo construido, asi que el tipo va PRIMERO
        # y la base despues: al reves, _rebuild_param_fields borraria lo que
        # acabamos de traer.
        self.structure_var.set(STRUCTURES[IMPORTED_KIND].label)
        self._rebuild_param_fields()
        self._imported_base = atoms.copy()
        self._on_built(atoms, origin=current.origin)
        self.status_var.set(
            f"Traída de «{current.origin}»: {len(self.atoms)} átomos. "
            "El tipo pasó a «Personalizada / importada», así que construir "
            "le aplica dopaje y grupos en vez de generar otra cosa.")

    def _discard_current_structure(self) -> None:
        """Drop the built structure and disable the actions that consume it."""
        # An EDLC cell built from this structure goes with it. Leaving it
        # exportable would write a cell whose electrode is no longer the one
        # on screen — the same trap the builder tab's export had.
        if self.edlc_cell is not None and self._edlc_source is self.atoms:
            self.edlc_cell = None
            self._edlc_source = None
            self.edlc_export_button.configure(state="disabled")
            self._set_edlc_report("")
            self.edlc_status_var.set(
                "La celda EDLC se descartó al cambiar la estructura."
            )
        # The window's current structure goes too, if it was this one:
        # otherwise Preparar would export a structure no longer on screen.
        session = getattr(self, "session", None)
        if session is not None and session.current is not None \
                and session.current is getattr(self, "_published", None):
            session.clear()
        self.atoms = None
        self.png_button.configure(state="disabled")
        # Puede no existir todavia: descartar corre tambien mientras se
        # construye la pestana, antes de que esten todos los botones.
        save = getattr(self, "save_button", None)
        if save is not None:
            save.configure(state="disabled")
        self.axes.clear()
        self.axes.set_title("Pulsa «Construir y previsualizar»")
        self.canvas.draw_idle()
        self._set_info("")
        self.status_var.set("Listo.")

    # ------------------------------------------------------------------
    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.build_button.configure(state=state)
        if message:
            self.status_var.set(message)
        if busy:
            self._build_started = time.monotonic()
            self._tick_clock()
        else:
            if self._clock_job is not None:
                self.root.after_cancel(self._clock_job)
                self._clock_job = None
            if self._build_started is not None:
                self.elapsed_var.set(f"tardó {_clock(time.monotonic() - self._build_started)}")
            self._build_started = None

    def _tick_clock(self) -> None:
        """Cuenta mientras se construye.

        Sin esto la ventana no dice nada durante una construcción: no hay
        barra de progreso, y un barrido de convergencia o una celda EDLC
        de varios miles de átomos se parecen mucho a un programa colgado.
        """
        if self._build_started is None:
            return
        self.elapsed_var.set(
            f"construyendo… {_clock(time.monotonic() - self._build_started)}")
        self._clock_job = self.root.after(250, self._tick_clock)

    def _on_build(self) -> None:
        if self._busy:
            return
        kind = self._current_structure_key()
        raw_params = self._read_raw(self._param_vars)
        raw_mods = self._read_raw(self._modifier_vars)
        raw_groups = self._read_raw(self._functionalization_vars)
        supports_mods = STRUCTURES[kind].supports_modifiers
        if kind == IMPORTED_KIND and self._imported_base is None:
            self.status_var.set(
                "No hay estructura que personalizar: impórtala en la página "
                "Importar, o pulsa «Traer la estructura actual».")
            return

        self._set_busy(True, "Construyendo… (puede tardar en estructuras grandes)")

        base = self._imported_base

        def worker() -> None:
            try:
                # Siempre desde la base original: construir dos veces no debe
                # dopar sobre lo ya dopado.
                atoms = base.copy() if kind == IMPORTED_KIND \
                    else build_structure(kind, raw_params)
                if supports_mods:
                    atoms = apply_modifiers(atoms, raw_mods)
                    # Groups go on after doping and defects: attaching first
                    # would decorate carbons a later vacancy removes.
                    atoms = apply_functionalization(
                        atoms, {**raw_mods, **raw_groups}
                    )
                self._queue.put(("built", atoms))
            except Exception as exc:  # surfaced to the user in a dialog
                self._queue.put(("error", (exc, traceback.format_exc())))

        threading.Thread(target=worker, daemon=True).start()

    def _on_built(self, atoms: Atoms, origin: str = "Construir") -> None:
        self.atoms = atoms
        session = getattr(self, "session", None)
        if session is not None:
            self._published = session.publish(atoms, origin)
        self._set_busy(False, f"Estructura lista: {len(atoms)} átomos.")
        self.png_button.configure(state="normal")
        save = getattr(self, "save_button", None)
        if save is not None:
            save.configure(state="normal")
        self._render(atoms)
        summary = describe_structure(atoms)
        # The structure can be geometrically perfect and still be a hopeless
        # request (Raman on a metal), so report both.
        try:
            physics = validate_calculation(atoms, self._all_values())
        except Exception as exc:  # never let the report break the preview
            physics = f"No se pudo evaluar el cálculo: {exc}"
        self._set_info(f"{summary}\n\n--- Cálculo solicitado (Preparar → Cálculo) ---\n"
                       f"{physics}")
        self._refresh_fixes(atoms)

    def _on_save_structure(self) -> None:
        """Escribe la geometría a un archivo, en el formato de la extensión.

        Se dice lo que la validación opina, pero no se impide guardar: un
        archivo de geometría es del usuario, y negarse a escribir lo que
        tiene en pantalla sólo le obliga a sacarlo por otro lado. Lo que sí
        pasa por validación es exportar un cálculo, que es otra cosa.
        """
        from tkinter import filedialog

        if self.atoms is None:
            self.status_var.set("No hay estructura que guardar: constrúyela "
                                "o tráela primero.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xyz",
            filetypes=[("extxyz", "*.xyz"), ("CIF", "*.cif"),
                       ("VASP POSCAR", "*.vasp"), ("PDB", "*.pdb"),
                       ("Todos", "*.*")])
        if not path:
            return
        try:
            clean = self.atoms.copy()
            # El info de los constructores lleva anillos y enlaces enteros, y
            # un extxyz con eso dentro no se vuelve a leer.
            clean.info = {}
            clean.write(path)
        except Exception as exc:                      # noqa: BLE001
            self.status_var.set(f"No se pudo guardar: {exc}")
            return
        note = ""
        try:
            report = run_basic_checks(self.atoms)
            if getattr(report, "errors", None):
                note = (f"  Ojo: la validación ve {len(report.errors)} "
                        "problema(s); míralos antes de calcular con ella.")
        except Exception:                             # noqa: BLE001
            pass
        self.status_var.set(f"Estructura guardada en {path}.{note}")

    def _set_info(self, text: str) -> None:
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")

