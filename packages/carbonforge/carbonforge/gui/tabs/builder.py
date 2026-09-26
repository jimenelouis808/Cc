"""The "Construir estructura" tab: parameters, fixes, build and export.

Structures are built on a worker thread; the host's queue brings the result
back to the Tk thread (``_on_built``). The fix panel and the advanced-parameter
dialog read the whole form through ``_all_values``.
"""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from ase import Atoms

from ..params import (
    ADVANCED_KEY,
    CALCULATION_PARAMS,
    PRESET_PARAMS,
    preview_preset,
    apply_fix,
    check_parameter_constraints,
    collect_fixes,
    FUNCTIONALIZATION_PARAMS,
    apply_functionalization,
    MODIFIER_PARAMS,
    STRUCTURES,
    apply_modifiers,
    build_structure,
    describe_structure,
    export_structure,
    validate_calculation,
)

_EXPORT_FORMATS = (
    ("qe", "Quantum ESPRESSO (pw.x / ph.x)"),
    ("siesta", "SIESTA (.fdf)"),
    ("lammps", "LAMMPS (data + input)"),
    ("xyz", "XYZ (visores: OVITO, VMD)"),
    ("cif", "CIF (VESTA)"),
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
    """Parameters on the left, preview on the right; build, check, fix, export.

    Uses from the host: ``tk``, ``ttk``, ``root``, ``atoms``, the ``_*_vars``
    stores, ``_queue``, ``_add_field``, ``_read_raw``, ``_show_error``, and the
    preview panel (``_build_preview``, ``_render``, ``_discard_current_structure``).
    """

    def _build_builder_tab(self, parent) -> None:
        tk, ttk = self.tk, self.ttk

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

        # Every other keyword QE and SIESTA accept, from their catalogues.
        self._advanced: dict[str, dict[str, Any]] = {}
        ttk.Button(
            actions, text="Parámetros avanzados (QE, SIESTA)…",
            command=self._on_advanced,
        ).pack(fill="x", pady=(4, 0))

        self.export_button = ttk.Button(
            actions, text="Exportar…", command=self._on_export, state="disabled"
        )
        self.export_button.pack(fill="x", pady=(4, 0))

        self.png_button = ttk.Button(
            actions, text="Guardar imagen PNG…", command=self._on_save_png,
            state="disabled",
        )
        self.png_button.pack(fill="x", pady=(4, 0))

        # Problems whose cure is a setting, each with a button that applies it.
        self.fix_frame = ttk.LabelFrame(left, text="Correcciones", padding=4)
        self.fix_frame.pack(fill="x", pady=(8, 0))
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
        self._calculation_vars.clear()
        self._preset_vars.clear()
        self._functionalization_vars.clear()
        self._format_vars.clear()

        # Switching structure type invalidates whatever was built before;
        # otherwise "Exportar" would silently write the previous structure.
        self._discard_current_structure()

        spec = STRUCTURES[self._current_structure_key()]
        self.description_label.configure(text=spec.description)

        # The recipe comes first: most people want a goal, not twenty knobs.
        recipe = ttk.LabelFrame(
            self.params_frame, text="1. ¿Qué quieres calcular?", padding=4
        )
        recipe.pack(fill="x")
        for param in PRESET_PARAMS:
            self._add_field(recipe, param, self._preset_vars)
        ttk.Button(
            recipe, text="Ver qué decidiría la receta",
            command=self._on_preview_preset,
        ).pack(fill="x", pady=(6, 0))

        geometry = ttk.LabelFrame(
            self.params_frame, text="2. Geometría", padding=4
        )
        geometry.pack(fill="x", pady=(10, 0))
        for param in spec.params:
            self._add_field(geometry, param, self._param_vars)

        if spec.supports_modifiers:
            mods = ttk.LabelFrame(
                self.params_frame, text="3. Dopaje y defectos", padding=4
            )
            mods.pack(fill="x", pady=(10, 0))
            for param in MODIFIER_PARAMS:
                self._add_field(mods, param, self._modifier_vars)

            groups = ttk.LabelFrame(
                self.params_frame,
                text="4. Grupos funcionales y nitrógeno",
                padding=4,
            )
            groups.pack(fill="x", pady=(10, 0))
            for param in FUNCTIONALIZATION_PARAMS:
                self._add_field(groups, param, self._functionalization_vars)

        calc = ttk.LabelFrame(
            self.params_frame,
            text="5. Ajustes manuales (se ignoran si usas una receta)",
            padding=4,
        )
        calc.pack(fill="x", pady=(10, 0))
        for param in CALCULATION_PARAMS:
            self._add_field(calc, param, self._calculation_vars)

        fmts = ttk.LabelFrame(
            self.params_frame, text="6. Formatos de salida", padding=4
        )
        fmts.pack(fill="x", pady=(10, 0))
        for key, label in _EXPORT_FORMATS:
            var = self.tk.BooleanVar(value=key in ("qe", "lammps"))
            ttk.Checkbutton(fmts, text=label, variable=var).pack(anchor="w")
            self._format_vars[key] = var

        self.force_var = self.tk.BooleanVar(value=False)
        ttk.Checkbutton(
            fmts,
            text="Exportar aunque falle la validación",
            variable=self.force_var,
        ).pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------------
    # Fixes
    # ------------------------------------------------------------------
    def _all_values(self) -> dict[str, Any]:
        return {
            **self._read_raw(self._param_vars),
            **self._read_raw(self._modifier_vars),
            **self._read_raw(self._functionalization_vars),
            **self._read_raw(self._calculation_vars),
            **self._read_raw(self._preset_vars),
            ADVANCED_KEY: self._advanced,
        }

    def _on_advanced(self) -> None:
        from ..advanced import AdvancedParamsDialog

        AdvancedParamsDialog(self.root, self._advanced, codes=("qe", "siesta"),
                             on_change=self._on_check_constraints)

    def _refresh_fixes(self, atoms: Optional[Atoms]) -> None:
        """Rebuild the fix panel from the current form and structure."""
        ttk = self.ttk
        for child in self.fix_frame.winfo_children():
            child.destroy()
        try:
            fixes = collect_fixes(atoms, self._all_values())
        except Exception as exc:  # the panel must never break the window
            ttk.Label(self.fix_frame, text=f"No se pudieron evaluar: {exc}",
                      wraplength=330).pack(anchor="w")
            return
        if not fixes:
            ttk.Label(self.fix_frame, text="Nada que corregir.",
                      foreground="#0a6").pack(anchor="w")
            return
        for severity, message, fix in fixes:
            row = ttk.Frame(self.fix_frame)
            row.pack(fill="x", pady=1)
            mark = "ERROR" if severity == "error" else "AVISO"
            label = ttk.Label(row, text=f"{mark}: {fix.label}", wraplength=250,
                              foreground="#a33" if severity == "error" else "#a60")
            label.pack(side="left", fill="x", expand=True)
            label.bind("<Enter>", lambda _e, m=message: self.status_var.set(m[:300]))
            ttk.Button(row, text="Aplicar", width=8,
                       command=lambda fx=fix: self._apply_fixes([fx])).pack(side="right")
        ttk.Button(self.fix_frame, text="Aplicar todas",
                   command=lambda: self._apply_fixes([fx for _, _, fx in fixes])
                   ).pack(fill="x", pady=(4, 0))

    def _apply_fixes(self, fixes) -> None:
        """Set the fields the fixes name, then re-check."""
        values = self._all_values()
        stores = (self._calculation_vars, self._preset_vars, self._functionalization_vars,
                  self._modifier_vars, self._param_vars)
        structural = False
        for fix in fixes:
            try:
                values = apply_fix(values, fix)
            except KeyError as exc:
                self.status_var.set(str(exc))
                continue
            for store in stores:
                if fix.setting in store:
                    store[fix.setting].set(values[fix.setting])
                    structural |= store is not self._calculation_vars \
                        and store is not self._preset_vars
                    break
        if structural and self.atoms is not None:
            # A change to the structure itself: rebuild so the preview matches.
            self._on_build()
            return
        self._on_check_constraints()
        if self.atoms is not None:
            self._on_built(self.atoms)

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

    def _on_preview_preset(self) -> None:
        """Show what the chosen recipe would decide, before building anything.

        Building the structure first is the point: a recipe adapts to it, so
        the answer for a zigzag ribbon differs from an armchair one.
        """
        kind = self._current_structure_key()
        try:
            atoms = build_structure(kind, self._read_raw(self._param_vars))
            text = preview_preset(atoms, self._read_raw(self._preset_vars))
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return
        self._set_info(text)
        self.status_var.set("Vista previa de la receta (nada construido aún).")

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
        self.atoms = None
        self.export_button.configure(state="disabled")
        self.png_button.configure(state="disabled")
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

        self._set_busy(True, "Construyendo… (puede tardar en estructuras grandes)")

        def worker() -> None:
            try:
                atoms = build_structure(kind, raw_params)
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

    def _on_built(self, atoms: Atoms) -> None:
        self.atoms = atoms
        self._set_busy(False, f"Estructura lista: {len(atoms)} átomos.")
        self.export_button.configure(state="normal")
        self.png_button.configure(state="normal")
        self._render(atoms)
        summary = describe_structure(atoms)
        # The structure can be geometrically perfect and still be a hopeless
        # request (Raman on a metal), so report both.
        try:
            physics = validate_calculation(atoms, self._all_values())
        except Exception as exc:  # never let the report break the preview
            physics = f"No se pudo evaluar el cálculo: {exc}"
        self._set_info(f"{summary}\n\n--- Cálculo solicitado ---\n{physics}")
        self._refresh_fixes(atoms)

    def _set_info(self, text: str) -> None:
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _on_export(self) -> None:
        from tkinter import filedialog, messagebox

        if self.atoms is None:
            return
        formats = [key for key, var in self._format_vars.items() if var.get()]
        if not formats:
            messagebox.showwarning(
                "Sin formatos", "Marca al menos un formato de salida."
            )
            return

        outdir = filedialog.askdirectory(title="Carpeta de destino")
        if not outdir:
            return
        try:
            written = export_structure(
                self.atoms,
                Path(outdir),
                formats,
                force=bool(self.force_var.get()),
                calculation_values={
                    **self._read_raw(self._calculation_vars),
                    **self._read_raw(self._preset_vars),
                    ADVANCED_KEY: self._advanced,
                },
            )
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return

        listado = "\n".join(f"  • {p}" for p in written)
        self.status_var.set(f"Exportado: {len(written)} archivo(s).")
        messagebox.showinfo("Exportación completada", f"Archivos escritos:\n{listado}")
