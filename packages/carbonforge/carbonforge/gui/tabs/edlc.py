"""The "Celda EDLC (LAMMPS)" tab: electrode + electrolyte cell for constant-potential MD.

The electrode is the window's current structure (built, imported or
modelled); a cell built from it is dropped when the current structure changes.
Packing is slow, so the cell is assembled on a worker thread and delivered
through the host's queue (``_on_edlc_built``).
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

from ase import Atoms

from ..edlc_params import (
    EDLC_PARAMS,
    build_edlc,
    check_edlc_constraints,
    describe_edlc,
    estimate_size,
    export_edlc,
)


class EdlcTab:
    """Assemble, check and export an EDLC cell.

    Uses from the host: ``tk``, ``ttk``, ``atoms``, ``_imported_atoms``,
    ``_queue``, ``_busy``, ``_add_field``, ``_read_raw``, ``_show_error``.
    """

    # ------------------------------------------------------------------
    # EDLC tab
    # ------------------------------------------------------------------
    def _build_edlc_tab(self, parent) -> None:
        """Turn the current structure into a constant-potential EDLC cell."""
        tk, ttk = self.tk, self.ttk

        outer = ttk.Frame(parent, padding=8)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer, width=400)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)
        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        intro = ttk.LabelFrame(left, text="Qué hace esta pestaña", padding=6)
        intro.pack(fill="x")
        ttk.Label(
            intro,
            text=(
                "Coloca la estructura de la primera pestaña como electrodo, "
                "la duplica enfrente y rellena el hueco con electrolito. Los "
                "electrodos se mantienen a potencial fijo (±V/2) y su carga "
                "responde: eso es lo que hace medible la capacitancia.\n\n"
                "Necesita el paquete ELECTRODE de LAMMPS, que no viene "
                "compilado por defecto."
            ),
            wraplength=370, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w")

        # Scrollable parameter column: there are more knobs than fit.
        params_box = ttk.LabelFrame(left, text="Parámetros", padding=4)
        params_box.pack(fill="both", expand=True, pady=(10, 0))
        canvas = tk.Canvas(params_box, highlightthickness=0, width=370)
        scroll = ttk.Scrollbar(
            params_box, orient="vertical", command=canvas.yview
        )
        frame = ttk.Frame(canvas)
        frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        for spec in EDLC_PARAMS:
            self._add_field(frame, spec, self._edlc_vars)

        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=(8, 0))
        ttk.Button(
            actions, text="Comprobar parámetros",
            command=self._on_edlc_check,
        ).pack(fill="x", pady=2)
        self.edlc_build_button = ttk.Button(
            actions, text="Construir celda EDLC",
            command=self._on_edlc_build,
        )
        self.edlc_build_button.pack(fill="x", pady=2)
        self.edlc_export_button = ttk.Button(
            actions, text="Exportar a LAMMPS…", state="disabled",
            command=self._on_edlc_export,
        )
        self.edlc_export_button.pack(fill="x", pady=2)

        self.edlc_status_var = tk.StringVar(
            value="Usa la estructura actual: constrúyela, impórtala o arma un modelo."
        )
        ttk.Label(
            left, textvariable=self.edlc_status_var, wraplength=380,
            justify="left", foreground="#0a6",
        ).pack(fill="x", pady=(8, 0))

        report_box = ttk.LabelFrame(right, text="Informe", padding=4)
        report_box.pack(fill="both", expand=True)
        self.edlc_text = tk.Text(report_box, wrap="word")
        report_scroll = ttk.Scrollbar(
            report_box, orient="vertical", command=self.edlc_text.yview
        )
        self.edlc_text.configure(
            yscrollcommand=report_scroll.set, state="disabled"
        )
        self.edlc_text.pack(side="left", fill="both", expand=True)
        report_scroll.pack(side="right", fill="y")

        session = getattr(self, "session", None)
        if session is not None:
            session.subscribe(self._on_edlc_session)

    def _on_edlc_session(self, current) -> None:
        """Drop a cell whose electrode is no longer the current structure."""
        source = getattr(self, "_edlc_source", None)
        if self.edlc_cell is not None and source is not None and source is not current \
                and not isinstance(source, Atoms):
            self.edlc_cell = None
            self._edlc_source = None
            self.edlc_export_button.configure(state="disabled")
            self._set_edlc_report("")
            self.edlc_status_var.set("La celda EDLC se descartó al cambiar la estructura actual.")

    def _set_edlc_report(self, text: str) -> None:
        self.edlc_text.configure(state="normal")
        self.edlc_text.delete("1.0", "end")
        self.edlc_text.insert("1.0", text)
        self.edlc_text.configure(state="disabled")

    def _edlc_electrode(self):
        """The structure to use as electrode, or ``None`` with a message set."""
        session = getattr(self, "session", None)
        if session is not None and session.current is not None:
            self._edlc_pending_source = session.current
            return session.take()
        self._edlc_pending_source = None
        atoms = self.atoms or getattr(self, "_imported_atoms", None)
        if atoms is None:
            self.edlc_status_var.set(
                "No hay estructura actual: constrúyela, impórtala o arma un modelo."
            )
        return atoms

    def _on_edlc_check(self) -> None:
        """Report the problems before paying to pack thousands of molecules."""
        atoms = self._edlc_electrode()
        if atoms is None:
            return
        raw = self._read_raw(self._edlc_vars)
        try:
            report = check_edlc_constraints(atoms, raw)
            size = estimate_size(atoms, raw)
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return
        self._set_edlc_report(f"{size}\n\n{report}")
        self.edlc_status_var.set("Parámetros comprobados (nada construido).")

    def _on_edlc_build(self) -> None:
        """Assemble the cell on a worker thread — packing is slow."""
        if self._busy:
            return
        atoms = self._edlc_electrode()
        if atoms is None:
            return
        raw = self._read_raw(self._edlc_vars)

        self.edlc_build_button.configure(state="disabled")
        self.edlc_status_var.set("Rellenando el electrolito… puede tardar.")

        def worker() -> None:
            try:
                self._queue.put(("edlc", build_edlc(atoms, raw)))
            except Exception as exc:
                self._queue.put(("edlc_error", (exc, traceback.format_exc())))

        threading.Thread(target=worker, daemon=True).start()

    def _on_edlc_built(self, cell) -> None:
        self.edlc_cell = cell
        self._edlc_source = getattr(self, "_edlc_pending_source", None) or (
            self.atoms or getattr(self, "_imported_atoms", None))
        self.edlc_build_button.configure(state="normal")
        self.edlc_export_button.configure(state="normal")
        self._set_edlc_report(describe_edlc(cell))
        self.edlc_status_var.set(f"Celda lista: {len(cell.atoms)} átomos.")

    def _on_edlc_export(self) -> None:
        cell = getattr(self, "edlc_cell", None)
        if cell is None:
            return

        from tkinter import filedialog, messagebox

        outdir = filedialog.askdirectory(title="Carpeta de destino")
        if not outdir:
            return
        try:
            written = export_edlc(
                cell, Path(outdir), self._read_raw(self._edlc_vars)
            )
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return
        listado = "\n".join(f"  • {p}" for p in written)
        self.edlc_status_var.set(f"Exportado: {len(written)} archivo(s).")
        messagebox.showinfo(
            "Exportación completada",
            f"Archivos escritos:\n{listado}\n\n"
            "Lee NOTAS_EDLC.txt antes de lanzar: explica qué paquete de "
            "LAMMPS hace falta y cómo sacar la capacitancia.",
        )
