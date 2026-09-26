"""Advanced parameters: every keyword a code accepts, searchable and checked.

The main form holds the settings carbonforge decides from the structure.
Everything else a code offers is reached here: pick the code, search its
catalogue (the curated core, or the full manual once imported), read what a
keyword does, give it a value. Values are checked against the catalogue as
they are entered, and again before anything is written.

The dialog edits a plain ``{code: {name: raw value}}`` dict owned by the
caller; the decisions (catalogue loading, checking, importing a manual) live
in functions here that need no display, so they can be tested headless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from ..codes import CODES, Catalog, Parameter, load_catalog, save_imported
from ..codes.importers import import_manual, introspect_gpaw

#: What to ask for when importing each code's documentation.
MANUAL_HINTS = {
    "qe": "INPUT_PW.def (en PW/Doc/ de las fuentes de Quantum ESPRESSO)",
    "siesta": "siesta.tex (en Docs/ de las fuentes de SIESTA)",
    "gpaw": "se lee del GPAW instalado; no hace falta archivo",
}


def describe_parameter(parameter: Parameter) -> str:
    """A readable card for one keyword: where it goes, type, default, options."""
    where = f"&{parameter.section.upper()}" if parameter.section else CODES[parameter.code]
    lines = [f"{parameter.name}   ({where})",
             f"Tipo: {parameter.kind}" + (f"   Unidades: {parameter.units}"
                                          if parameter.units else "")]
    if parameter.default:
        lines.append(f"Por defecto: {parameter.default}")
    if parameter.choices:
        lines.append("Opciones: " + ", ".join(parameter.choices))
    if parameter.managed:
        lines.append("carbonforge lo decide a partir de la estructura y los controles; "
                     "un valor aquí lo sustituye.")
    lines.append(f"Fuente: {parameter.source}")
    if parameter.description:
        lines += ["", parameter.description]
    return "\n".join(lines)


def import_documentation(code: str, path: Optional[str | Path] = None) -> tuple[Catalog, Path]:
    """Import a code's documentation into the user catalogue.

    QE and SIESTA need their manual file; GPAW is read from the installed
    package (``path`` ignored). Returns the merged catalogue and where the
    import was stored.
    """
    if code == "gpaw":
        imported = introspect_gpaw()
    else:
        if path is None:
            raise ValueError(f"Hace falta el archivo del manual: {MANUAL_HINTS[code]}.")
        imported = import_manual(path)
        if imported.code != code:
            raise ValueError(
                f"Ese archivo es del manual de {CODES[imported.code]}, no de {CODES[code]}."
            )
    stored = save_imported(imported)
    return load_catalog(code), stored


def check_overrides(code: str, overrides: dict[str, Any],
                    catalog: Optional[Catalog] = None) -> tuple[dict[str, Any], list[str]]:
    """Typed overrides and the messages to show (errors first)."""
    catalog = catalog or load_catalog(code)
    typed, report = catalog.check(overrides)
    return typed, [f"ERROR: {e}" for e in report.errors] + [f"AVISO: {w}"
                                                           for w in report.warnings]


class AdvancedParamsDialog:
    """A window editing ``overrides[code]`` for the given codes.

    Parameters
    ----------
    parent
        The Tk widget that owns the window.
    overrides
        ``{code: {name: raw}}``, edited in place.
    codes
        Which codes can be chosen (``("qe", "siesta")`` for the builder,
        ``("gpaw",)`` for vibspec).
    on_change
        Called after every change, e.g. to re-run the checks.
    """

    def __init__(self, parent, overrides: dict[str, dict[str, Any]],
                 codes: tuple[str, ...] = ("qe", "siesta", "gpaw"),
                 on_change: Optional[Callable[[], None]] = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.overrides = overrides
        self.on_change = on_change
        self.codes = codes
        self._catalogs: dict[str, Catalog] = {}
        self._shown: list[Parameter] = []

        win = self.window = tk.Toplevel(parent)
        win.title("Parámetros avanzados")
        win.geometry("980x620")

        top = ttk.Frame(win, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="Código").pack(side="left")
        self.code_var = tk.StringVar(value=codes[0])
        code_box = ttk.Combobox(top, textvariable=self.code_var, state="readonly", width=24,
                                values=list(codes))
        code_box.pack(side="left", padx=4)
        code_box.bind("<<ComboboxSelected>>", lambda _e: self._on_code())
        ttk.Label(top, text="Buscar").pack(side="left", padx=(12, 0))
        self.search_var = tk.StringVar()
        entry = ttk.Entry(top, textvariable=self.search_var, width=28)
        entry.pack(side="left", padx=4)
        self.search_var.trace_add("write", lambda *_: self._refresh_list())
        ttk.Button(top, text="Importar manual…", command=self._on_import).pack(side="right")
        self.source_var = tk.StringVar()
        ttk.Label(top, textvariable=self.source_var, foreground="#667").pack(side="right", padx=8)

        body = ttk.Panedwindow(win, orient="horizontal")
        body.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(body)
        body.add(left, weight=2)
        columns = ("seccion", "tipo", "defecto")
        self.tree = ttk.Treeview(left, columns=columns, show="tree headings", height=20)
        self.tree.heading("#0", text="parámetro")
        self.tree.heading("seccion", text="sección")
        self.tree.heading("tipo", text="tipo")
        self.tree.heading("defecto", text="por defecto")
        self.tree.column("#0", width=220)
        self.tree.column("seccion", width=80)
        self.tree.column("tipo", width=70)
        self.tree.column("defecto", width=120)
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())

        right = ttk.Frame(body)
        body.add(right, weight=3)
        self.card = tk.Text(right, height=12, wrap="word", font="TkTextFont")
        self.card.pack(fill="both", expand=True)
        self.card.configure(state="disabled")

        edit = ttk.Frame(right)
        edit.pack(fill="x", pady=4)
        ttk.Label(edit, text="Valor").pack(side="left")
        self.value_var = tk.StringVar()
        self.value_box = ttk.Combobox(edit, textvariable=self.value_var, width=36)
        self.value_box.pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(edit, text="Añadir / actualizar", command=self._on_set).pack(side="left")

        ttk.Label(right, text="Parámetros avanzados de este código").pack(anchor="w")
        self.chosen = tk.Listbox(right, height=7)
        self.chosen.pack(fill="x")
        self.chosen.bind("<<ListboxSelect>>", lambda _e: self._on_pick_chosen())
        ttk.Button(right, text="Quitar seleccionado", command=self._on_remove).pack(anchor="e")
        self.messages = tk.Text(right, height=6, wrap="word", foreground="#a33",
                                font="TkTextFont")
        self.messages.pack(fill="x", pady=(4, 0))
        self.messages.configure(state="disabled")

        self._on_code()

    # -- helpers ------------------------------------------------------------

    @property
    def code(self) -> str:
        return self.code_var.get()

    def catalog(self) -> Catalog:
        if self.code not in self._catalogs:
            self._catalogs[self.code] = load_catalog(self.code)
        return self._catalogs[self.code]

    def _set_text(self, widget, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _current(self) -> dict[str, Any]:
        return self.overrides.setdefault(self.code, {})

    # -- events -------------------------------------------------------------

    def _on_code(self) -> None:
        catalog = self.catalog()
        origin = "manual importado" if catalog.documented else "solo núcleo curado"
        self.source_var.set(f"{len(catalog.parameters)} parámetros ({origin})")
        self._refresh_list()
        self._refresh_chosen()

    def _refresh_list(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self._shown = self.catalog().search(self.search_var.get())
        for k, parameter in enumerate(self._shown):
            name = ("• " if parameter.managed else "") + parameter.name
            self.tree.insert("", "end", iid=str(k), text=name,
                             values=(parameter.section, parameter.kind, parameter.default or ""))

    def _selected(self) -> Optional[Parameter]:
        selection = self.tree.selection()
        return self._shown[int(selection[0])] if selection else None

    def _on_select(self) -> None:
        parameter = self._selected()
        if parameter is None:
            return
        self._set_text(self.card, describe_parameter(parameter))
        self.value_box.configure(values=list(parameter.choices) or (
            ["true", "false"] if parameter.kind == "logical" else []))
        self.value_var.set(str(self._current().get(parameter.qualified, "")))

    def _refresh_chosen(self) -> None:
        self.chosen.delete(0, "end")
        current = self._current()
        for name, value in current.items():
            self.chosen.insert("end", f"{name} = {value}")
        _, messages = check_overrides(self.code, current, self.catalog())
        self._set_text(self.messages, "\n".join(messages) or "Sin problemas.")

    def _changed(self) -> None:
        self._refresh_chosen()
        if self.on_change is not None:
            self.on_change()

    def _on_set(self) -> None:
        parameter = self._selected()
        if parameter is None:
            self._set_text(self.messages, "Elige un parámetro de la lista.")
            return
        value = self.value_var.get().strip()
        if not value:
            return
        try:
            parameter.coerce(value)
        except ValueError as exc:
            self._set_text(self.messages, f"ERROR: {exc}")
            return
        self._current()[parameter.qualified] = value
        self._changed()

    def _on_pick_chosen(self) -> None:
        selection = self.chosen.curselection()
        if not selection:
            return
        name = self.chosen.get(selection[0]).split(" = ", 1)[0]
        self.search_var.set(name.rpartition(".")[2] if self.code == "qe" else name)

    def _on_remove(self) -> None:
        selection = self.chosen.curselection()
        if not selection:
            return
        name = self.chosen.get(selection[0]).split(" = ", 1)[0]
        self._current().pop(name, None)
        self._changed()

    def _on_import(self) -> None:
        from tkinter import filedialog, messagebox

        path = None
        if self.code != "gpaw":
            path = filedialog.askopenfilename(
                parent=self.window, title=f"Manual: {MANUAL_HINTS[self.code]}",
                filetypes=[("Manual", "*.def *.tex"), ("Todos", "*")])
            if not path:
                return
        try:
            catalog, stored = import_documentation(self.code, path)
        except (ImportError, ValueError, OSError) as exc:
            messagebox.showerror("No se pudo importar", str(exc), parent=self.window)
            return
        self._catalogs[self.code] = catalog
        self._on_code()
        self._set_text(self.messages, f"Importado: {len(catalog.parameters)} parámetros; "
                                      f"guardado en {stored}.")
