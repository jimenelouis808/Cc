"""The "Importar y preparar" tab: read a structure file, repair it, scan pseudopotentials."""

from __future__ import annotations

import traceback


from ..params import (
    import_and_repair,
    scan_pseudopotentials,
)


class ImportTab:
    """Import and repair a structure; hand it to the builder.

    Uses from the host: ``tk``, ``ttk``, ``root``, ``_show_error`` and, to adopt
    the structure, the builder's ``_on_built``.
    """

    # ------------------------------------------------------------------
    # Import tab
    # ------------------------------------------------------------------
    def _build_import_tab(self, parent) -> None:
        """Bring in a structure from elsewhere, and find the files it needs."""
        tk, ttk = self.tk, self.ttk

        outer = ttk.Frame(parent, padding=8)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer, width=360)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)
        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        # --- structure import -------------------------------------------
        box = ttk.LabelFrame(left, text="Importar estructura", padding=6)
        box.pack(fill="x")
        ttk.Label(
            box,
            text=("CIF, XYZ, POSCAR, salidas de QE, datos de LAMMPS, PDB… "
                  "Se revisa lo que falta y se repara lo que se pueda."),
            wraplength=330, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w")

        self.autofix_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            box, text="Reparar automáticamente lo que sea seguro",
            variable=self.autofix_var,
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            box,
            text=("Añade la celda que falte, quita átomos duplicados, "
                  "repliega coordenadas y amplía el vacío. Nunca separa "
                  "átomos superpuestos: eso inventaría una estructura."),
            wraplength=330, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w")

        ttk.Button(
            box, text="Abrir archivo…", command=self._on_import_structure
        ).pack(fill="x", pady=(6, 0))

        self.use_imported_button = ttk.Button(
            box, text="Usar esta estructura", state="disabled",
            command=self._on_use_imported,
        )
        self.use_imported_button.pack(fill="x", pady=(4, 0))

        # --- pseudopotentials -------------------------------------------
        pseudo = ttk.LabelFrame(left, text="Pseudopotenciales", padding=6)
        pseudo.pack(fill="x", pady=(10, 0))
        ttk.Label(
            pseudo,
            text=("Lee la cabecera de cada UPF, no su nombre, así que "
                  "distingue de verdad NC de PAW y escalar de relativista."),
            wraplength=330, justify="left", foreground="#777777",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w")

        self.pseudo_raman_var = tk.BooleanVar(value=False)
        self.pseudo_soc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            pseudo, text="Para Raman (exige norm-conserving)",
            variable=self.pseudo_raman_var,
        ).pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(
            pseudo, text="Para espín-órbita (exige relativistas)",
            variable=self.pseudo_soc_var,
        ).pack(anchor="w")

        ttk.Button(
            pseudo, text="Escanear carpeta…", command=self._on_scan_pseudos
        ).pack(fill="x", pady=(6, 0))

        self.import_status_var = tk.StringVar(value="Nada importado aún.")
        ttk.Label(
            left, textvariable=self.import_status_var, wraplength=340,
            justify="left", foreground="#0a6",
        ).pack(fill="x", pady=(10, 0))

        # --- report ------------------------------------------------------
        report_box = ttk.LabelFrame(right, text="Informe", padding=4)
        report_box.pack(fill="both", expand=True)
        self.import_text = self.tk.Text(report_box, wrap="word")
        report_scroll = ttk.Scrollbar(
            report_box, orient="vertical", command=self.import_text.yview
        )
        self.import_text.configure(
            yscrollcommand=report_scroll.set, state="disabled"
        )
        self.import_text.pack(side="left", fill="both", expand=True)
        report_scroll.pack(side="right", fill="y")

    def _set_import_report(self, text: str) -> None:
        self.import_text.configure(state="normal")
        self.import_text.delete("1.0", "end")
        self.import_text.insert("1.0", text)
        self.import_text.configure(state="disabled")

    def _on_import_structure(self) -> None:
        from tkinter import filedialog

        from ...io import IMPORT_FORMATS

        patterns = " ".join(f"*{ext}" for ext in sorted(IMPORT_FORMATS))
        path = filedialog.askopenfilename(
            title="Importar estructura",
            filetypes=[
                ("Estructuras reconocidas", patterns),
                ("Cualquiera", "*"),
            ],
        )
        if not path:
            return
        try:
            atoms, report = import_and_repair(
                path, autofix_it=bool(self.autofix_var.get())
            )
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return

        self._imported_atoms = atoms
        self._set_import_report(report)
        self.use_imported_button.configure(state="normal")
        self.import_status_var.set(
            f"{len(atoms)} átomos importados. Pulsa «Usar esta estructura» "
            "para trabajar con ella."
        )

    def _on_use_imported(self) -> None:
        """Adopt the imported structure as the one being worked on."""
        atoms = getattr(self, "_imported_atoms", None)
        if atoms is None:
            return
        self._on_built(atoms)
        self.import_status_var.set(
            "Estructura adoptada: ya puedes exportarla o funcionalizarla "
            "desde la primera pestaña."
        )

    def _on_scan_pseudos(self) -> None:
        from tkinter import filedialog

        directory = filedialog.askdirectory(
            title="Carpeta de pseudopotenciales"
        )
        if not directory:
            return
        atoms = self.atoms or getattr(self, "_imported_atoms", None)
        try:
            report = scan_pseudopotentials(
                directory,
                atoms=atoms,
                needs_raman=bool(self.pseudo_raman_var.get()),
                needs_soc=bool(self.pseudo_soc_var.get()),
            )
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return
        self._set_import_report(report)
        self.import_status_var.set("Carpeta escaneada.")
