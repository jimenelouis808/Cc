"""The "Preparar → Cálculo (QE, SIESTA, LAMMPS)" page: from a structure to inputs.

It always prepares the window's **current structure** (built, imported or
modelled on another page), so a vibspec model or an imported file is exported
exactly like a built one. The recipe, the manual calculation settings, the
advanced parameters, the fix panel and the export live here; the geometry
lives on the Construir page.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Optional

from ase import Atoms

from ..params import (
    ADVANCED_KEY,
    CALCULATION_PARAMS,
    PRESET_PARAMS,
    apply_fix,
    build_structure,
    check_parameter_constraints,
    collect_fixes,
    export_structure,
    preview_preset,
    validate_calculation,
)

EXPORT_FORMATS = (
    ("qe", "Quantum ESPRESSO (pw.x / ph.x)"),
    ("siesta", "SIESTA (.fdf)"),
    ("lammps", "LAMMPS (data + input)"),
    ("xyz", "XYZ (visores: OVITO, VMD)"),
    ("cif", "CIF (VESTA)"),
)


class PrepareTab:
    """Recipe, calculation settings, fixes and export for the current structure.

    Uses from the host: ``tk``, ``ttk``, ``root``, ``session``, the
    ``_*_vars`` stores, ``_add_field``, ``_read_raw``, ``_all_values``,
    ``_show_error``; and from the Construir page, for fixes that change the
    geometry, ``_on_build``, ``_current_structure_key`` and ``_published``.
    """

    def _build_prepare_tab(self, parent) -> None:
        tk, ttk = self.tk, self.ttk
        self._advanced: dict[str, dict[str, Any]] = {}

        outer = ttk.Frame(parent, padding=8)
        outer.pack(fill="both", expand=True)
        left = ttk.Frame(outer, width=380)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)
        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        which = ttk.LabelFrame(left, text="Se prepara", padding=6)
        which.pack(fill="x")
        self.prepare_structure_var = tk.StringVar(
            value="Nada todavía: construye, importa o arma un modelo.")
        ttk.Label(which, textvariable=self.prepare_structure_var, wraplength=340,
                  justify="left").pack(anchor="w")

        box = ttk.LabelFrame(left, text="Cálculo", padding=4)
        box.pack(fill="both", expand=True, pady=(8, 0))
        canvas = tk.Canvas(box, highlightthickness=0, width=340)
        scroll = ttk.Scrollbar(box, orient="vertical", command=canvas.yview)
        form = ttk.Frame(canvas)
        form.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # The recipe comes first: most people want a goal, not twenty knobs.
        recipe = ttk.LabelFrame(form, text="1. ¿Qué quieres calcular?", padding=4)
        recipe.pack(fill="x")
        for param in PRESET_PARAMS:
            self._add_field(recipe, param, self._preset_vars)
        ttk.Button(recipe, text="Ver qué decidiría la receta",
                   command=self._on_preview_preset).pack(fill="x", pady=(6, 0))

        manual = ttk.LabelFrame(form, text="2. Ajustes manuales (se ignoran si usas una receta)",
                                padding=4)
        manual.pack(fill="x", pady=(10, 0))
        for param in CALCULATION_PARAMS:
            self._add_field(manual, param, self._calculation_vars)

        fmts = ttk.LabelFrame(form, text="3. Formatos de salida", padding=4)
        fmts.pack(fill="x", pady=(10, 0))
        for key, label in EXPORT_FORMATS:
            var = tk.BooleanVar(value=key in ("qe", "lammps"))
            ttk.Checkbutton(fmts, text=label, variable=var).pack(anchor="w")
            self._format_vars[key] = var
        self.force_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fmts, text="Exportar aunque falle la validación",
                        variable=self.force_var).pack(anchor="w", pady=(4, 0))
        # Run it from here when the program is installed on this machine.
        self.enqueue_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fmts, text="Encolar al exportar (Calcular → Trabajos)",
                        variable=self.enqueue_var).pack(anchor="w")

        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=(8, 0))
        ttk.Button(actions, text="Comprobar el cálculo",
                   command=self._on_check_prepare).pack(fill="x")
        # Every other keyword QE and SIESTA accept, from their catalogues.
        ttk.Button(actions, text="Parámetros avanzados (QE, SIESTA)…",
                   command=self._on_advanced).pack(fill="x", pady=(4, 0))
        self.export_button = ttk.Button(actions, text="Exportar…", command=self._on_export,
                                        state="disabled")
        self.export_button.pack(fill="x", pady=(4, 0))

        # Problems whose cure is a setting, each with a button that applies it.
        self.prepare_fix_frame = ttk.LabelFrame(left, text="Correcciones", padding=4)
        self.prepare_fix_frame.pack(fill="x", pady=(8, 0))
        self._fix_frames.append(self.prepare_fix_frame)
        ttk.Label(self.prepare_fix_frame, text="Comprueba para ver qué se puede corregir.",
                  foreground="#777777", wraplength=330).pack(anchor="w")

        self.prepare_status_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.prepare_status_var, wraplength=340,
                  justify="left", foreground="#0a6").pack(fill="x", pady=(8, 0))

        report = ttk.LabelFrame(right, text="Informe del cálculo", padding=4)
        report.pack(fill="both", expand=True)
        self.prepare_text = tk.Text(report, wrap="word")
        bar = ttk.Scrollbar(report, orient="vertical", command=self.prepare_text.yview)
        self.prepare_text.configure(yscrollcommand=bar.set, state="disabled")
        bar.pack(side="right", fill="y")
        self.prepare_text.pack(side="left", fill="both", expand=True)

        session = getattr(self, "session", None)
        if session is not None:
            session.subscribe(self._on_prepare_structure)

    # -- state ---------------------------------------------------------

    def _prepared_atoms(self) -> Optional[Atoms]:
        """What gets prepared: the window's current structure (a copy)."""
        session = getattr(self, "session", None)
        if session is not None:
            return session.take()
        return getattr(self, "atoms", None)

    def _on_prepare_structure(self, current) -> None:
        if current is None:
            self.prepare_structure_var.set("Nada todavía: construye, importa o arma un modelo.")
            self.export_button.configure(state="disabled")
            return
        self.prepare_structure_var.set(current.describe())
        self.export_button.configure(state="normal")

    def _set_prepare_report(self, text: str) -> None:
        self.prepare_text.configure(state="normal")
        self.prepare_text.delete("1.0", "end")
        self.prepare_text.insert("1.0", text)
        self.prepare_text.configure(state="disabled")

    def _calculation_values(self) -> dict[str, Any]:
        return {**self._read_raw(self._calculation_vars), **self._read_raw(self._preset_vars),
                ADVANCED_KEY: self._advanced}

    # -- actions -------------------------------------------------------

    def _on_check_prepare(self) -> None:
        """Parameter rules and the physics of the calculation, for the current structure."""
        atoms = self._prepared_atoms()
        values = self._all_values()
        text = check_parameter_constraints(atoms, values)
        if atoms is None:
            text += "\n\n(Sin estructura actual: solo se comprobaron los parámetros.)"
        else:
            try:
                physics = validate_calculation(atoms, values)
            except Exception as exc:  # never let the report break the page
                physics = f"No se pudo evaluar el cálculo: {exc}"
            text += f"\n\n--- Cálculo solicitado ---\n{physics}"
        self._set_prepare_report(text)
        self._refresh_fixes(atoms)
        self.prepare_status_var.set("Cálculo comprobado.")

    def _on_advanced(self) -> None:
        from ..advanced import AdvancedParamsDialog

        AdvancedParamsDialog(self.root, self._advanced, codes=("qe", "siesta"),
                             on_change=self._on_check_prepare)

    def _on_preview_preset(self) -> None:
        """Show what the chosen recipe would decide, before writing anything.

        A recipe adapts to the structure (a zigzag ribbon gets edge spin), so
        it is previewed on the current structure, or, without one, on what
        the Construir form would build.
        """
        try:
            atoms = self._prepared_atoms()
            if atoms is None:
                atoms = build_structure(self._current_structure_key(),
                                        self._read_raw(self._param_vars))
            text = preview_preset(atoms, self._read_raw(self._preset_vars))
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return
        self._set_prepare_report(text)
        self.prepare_status_var.set("Vista previa de la receta (nada escrito aún).")

    def _on_export(self) -> None:
        from tkinter import filedialog, messagebox

        atoms = self._prepared_atoms()
        if atoms is None:
            return
        formats = [key for key, var in self._format_vars.items() if var.get()]
        if not formats:
            messagebox.showwarning("Sin formatos", "Marca al menos un formato de salida.")
            return
        outdir = filedialog.askdirectory(title="Carpeta de destino")
        if not outdir:
            return
        try:
            written = export_structure(atoms, Path(outdir), formats,
                                       force=bool(self.force_var.get()),
                                       calculation_values=self._calculation_values())
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return
        listado = "\n".join(f"  • {p}" for p in written)
        self.prepare_status_var.set(f"Exportado: {len(written)} archivo(s).")
        if getattr(self, "enqueue_var", None) is not None and self.enqueue_var.get():
            self._enqueue_exported(written)
        messagebox.showinfo("Exportación completada", f"Archivos escritos:\n{listado}")

    def _enqueue_exported(self, written) -> None:
        """Queue each exported engine directory (those with a job.json)."""
        from ...jobs.manifest import MANIFEST

        directories = [Path(p).parent for p in written if Path(p).name == MANIFEST]
        queued = [d.name for d in directories if self.submit_job(d) is not None]
        skipped = len(directories) - len(queued)
        note = f"Encolado: {', '.join(queued)}." if queued else "Nada encolado."
        if skipped:
            note += (f" {skipped} sin encolar: falta el programa aquí (ver Calcular → "
                     "Trabajos).")
        self.prepare_status_var.set(note)

    # -- fixes (shown on Construir and here) -----------------------------

    def _refresh_fixes(self, atoms: Optional[Atoms] = None) -> None:
        """Rebuild every fix panel from the form and the current structure."""
        if atoms is None:
            atoms = self._prepared_atoms()
        try:
            fixes, error = collect_fixes(atoms, self._all_values()), None
        except Exception as exc:  # the panel must never break the window
            fixes, error = [], exc
        for frame in self._fix_frames:
            self._render_fixes(frame, fixes, error)

    def _render_fixes(self, frame, fixes, error) -> None:
        ttk = self.ttk
        for child in frame.winfo_children():
            child.destroy()
        if error is not None:
            ttk.Label(frame, text=f"No se pudieron evaluar: {error}",
                      wraplength=330).pack(anchor="w")
            return
        if not fixes:
            ttk.Label(frame, text="Nada que corregir.", foreground="#0a6").pack(anchor="w")
            return
        for severity, message, fix in fixes:
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=1)
            mark = "ERROR" if severity == "error" else "AVISO"
            label = ttk.Label(row, text=f"{mark}: {fix.label}", wraplength=250,
                              foreground="#a33" if severity == "error" else "#a60")
            label.pack(side="left", fill="x", expand=True)
            label.bind("<Enter>", lambda _e, m=message: self.prepare_status_var.set(m[:300]))
            ttk.Button(row, text="Aplicar", width=8,
                       command=lambda fx=fix: self._apply_fixes([fx])).pack(side="right")
        ttk.Button(frame, text="Aplicar todas",
                   command=lambda: self._apply_fixes([fx for _, _, fx in fixes])
                   ).pack(fill="x", pady=(4, 0))

    def _apply_fixes(self, fixes) -> None:
        """Set the fields the fixes name, then re-check.

        A fix of the geometry (group count, vacuum) changes the Construir
        form; the structure is rebuilt only when the current structure is
        the one Construir built, since an imported one has no form to redo.
        """
        values = self._all_values()
        stores = (self._calculation_vars, self._preset_vars, self._functionalization_vars,
                  self._modifier_vars, self._param_vars)
        structural = False
        for fix in fixes:
            try:
                values = apply_fix(values, fix)
            except KeyError as exc:
                self.prepare_status_var.set(str(exc))
                continue
            for store in stores:
                if fix.setting in store:
                    store[fix.setting].set(values[fix.setting])
                    structural |= store is not self._calculation_vars \
                        and store is not self._preset_vars
                    break
        session = getattr(self, "session", None)
        built_here = session is None or (
            session.current is not None and session.current is getattr(self, "_published", None))
        if structural and getattr(self, "atoms", None) is not None and built_here:
            # A change to the structure itself: rebuild so the preview matches.
            self._on_build()
            return
        if structural and not built_here:
            self.prepare_status_var.set(
                "Corrección aplicada al formulario de Construir; la estructura actual no "
                "salió de ahí, así que no se reconstruye.")
        self._on_check_prepare()
