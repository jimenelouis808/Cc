"""El editor de recetas: los paneles salen del catálogo, no del código.

Cada tipo de cálculo describe sus parámetros en :mod:`carbonforge.dft.kinds`
-- tipo, rango, valor por defecto y ayuda --, y este panel los pinta
leyendo eso. Añadir un cálculo nuevo no toca esta pestaña.

Tres cosas que lo separan de un formulario:

* **La ayuda está donde se necesita.** Cada parámetro muestra la suya al
  seleccionarlo, en vez de obligar a buscarla en un manual.
* **Lo avanzado se esconde, pero no se quita.** Lo que casi nadie toca
  (base LCAO, ensanchamiento, criterio del SCF) queda detrás de una
  casilla; sigue siendo editable y sigue guardándose con la receta.
* **La validación corre mientras se escribe**, y es física: puntos k sobre
  un eje que no se repite, vacío insuficiente para ondas planas, un hueco
  de core sin átomo. Y la cadena se valida como cadena: un DOS sin estado
  base delante no cierra.

Delgado a propósito: lo que decide está en ``kinds`` y ``recipe``, que se
prueban sin Tk.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

from ase import Atoms

from .kinds import KINDS, Parameter, kind
from .recipe import Recipe, RecipeStep

__all__ = ["RecipePanel"]


class RecipePanel(ttk.Frame):
    """Compone una receta: elige cálculos, ajusta sus parámetros, valida."""

    def __init__(self, master, *, on_change: Optional[Callable] = None,
                 **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.on_change = on_change
        self.atoms: Optional[Atoms] = None
        self.steps: list[RecipeStep] = []
        self._vars: dict[str, tk.Variable] = {}
        self._rows: list[tuple] = []
        self.var_kind = tk.StringVar(value="relax")
        self.var_advanced = tk.BooleanVar(value=False)
        self.var_name = tk.StringVar(value="calculo")

        self._labels = {KINDS[k].label: k for k in KINDS}

        # Cabecera y listas primero: el area de parametros es la que se
        # expande, y lo que se empaqueta despues de algo con expand=True se
        # queda sin espacio.
        head = ttk.Frame(self)
        head.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(head, text="Cálculo:").pack(side="left")
        self.combo = ttk.Combobox(head, state="readonly", width=24,
                                  values=[KINDS[k].label for k in KINDS])
        self.combo.set(KINDS["relax"].label)
        self.combo.pack(side="left", padx=4)
        self.combo.bind("<<ComboboxSelected>>", lambda _e: self._on_kind())
        ttk.Checkbutton(head, text="avanzados", variable=self.var_advanced,
                        command=self._fill).pack(side="left", padx=8)
        ttk.Label(head, text="nombre:").pack(side="left")
        ttk.Entry(head, textvariable=self.var_name, width=14).pack(side="left")

        self.lbl_summary = ttk.Label(self, text="", wraplength=620,
                                     justify="left", foreground="#444444")
        self.lbl_summary.pack(fill="x", padx=6, pady=(4, 0))

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=6, pady=4)
        ttk.Button(buttons, text="Añadir a la receta",
                   command=self.add_step).pack(side="left")
        ttk.Button(buttons, text="Quitar el último",
                   command=self.remove_last).pack(side="left", padx=4)
        ttk.Button(buttons, text="Vaciar",
                   command=self.clear).pack(side="left")

        self.txt_problems = tk.Text(self, height=5, wrap="word", relief="flat",
                                    font=("TkFixedFont", 8))
        self.txt_problems.pack(fill="x", padx=6)
        self.txt_problems.configure(state="disabled")

        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=6, pady=(4, 6))

        left = ttk.Labelframe(panes, text="Parámetros", padding=4)
        canvas = tk.Canvas(left, highlightthickness=0, width=360)
        bar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        self.fields = ttk.Frame(canvas)
        self.fields.bind("<Configure>", lambda _e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.fields, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        panes.add(left, weight=3)

        right = ttk.Frame(panes)
        helpbox = ttk.Labelframe(right, text="Ayuda", padding=4)
        helpbox.pack(fill="both", expand=True)
        self.txt_help = tk.Text(helpbox, wrap="word", relief="flat", width=34,
                                font=("TkDefaultFont", 9))
        self.txt_help.pack(fill="both", expand=True)
        self.txt_help.configure(state="disabled")

        listbox = ttk.Labelframe(right, text="Receta", padding=4)
        listbox.pack(fill="both", expand=True, pady=(6, 0))
        self.tree = ttk.Treeview(listbox, columns=("n",), show="tree",
                                 height=7, selectmode="browse")
        self.tree.pack(fill="both", expand=True)
        panes.add(right, weight=2)

        self._on_kind()

    # ----------------------------------------------------------- estructura

    def set_structure(self, atoms: Optional[Atoms]) -> None:
        """La estructura contra la que se valida."""
        self.atoms = atoms
        self._validate()

    # ---------------------------------------------------------------- tipo

    @property
    def current_kind(self) -> str:
        return self._labels[self.combo.get()]

    def _on_kind(self) -> None:
        self.var_kind.set(self.current_kind)
        chosen = kind(self.current_kind)
        requires = (" Necesita antes: "
                    + ", ".join(KINDS[r].label for r in chosen.requires)
                    + "." if chosen.requires else "")
        self.lbl_summary.configure(text=chosen.summary + requires)
        self._show_help(chosen.help)
        self._fill()

    def _fill(self) -> None:
        """Pinta los campos del tipo elegido, leyendo el catálogo."""
        for child in self.fields.winfo_children():
            child.destroy()
        self._vars.clear()
        self._rows.clear()
        chosen = kind(self.current_kind)
        show_advanced = bool(self.var_advanced.get())

        for row, parameter in enumerate(chosen.parameters):
            if parameter.advanced and not show_advanced:
                continue
            self._add_field(row, parameter)
        self._validate()

    def _add_field(self, row: int, parameter: Parameter) -> None:
        unit = f" ({parameter.unit})" if parameter.unit else ""
        label = ttk.Label(self.fields, text=parameter.label + unit)
        label.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)

        if parameter.type == "bool":
            variable: tk.Variable = tk.BooleanVar(value=bool(parameter.default))
            widget = ttk.Checkbutton(self.fields, variable=variable,
                                     command=self._validate)
        elif parameter.type == "choice":
            variable = tk.StringVar(value=str(parameter.default))
            widget = ttk.Combobox(self.fields, textvariable=variable, width=16,
                                  values=list(parameter.choices),
                                  state="readonly")
            widget.bind("<<ComboboxSelected>>", lambda _e: self._validate())
        else:
            default = parameter.default
            text = ("" if parameter.type == "indices"
                    else str(default))
            variable = tk.StringVar(value=text)
            widget = ttk.Entry(self.fields, textvariable=variable, width=18)
            variable.trace_add("write", lambda *_a: self._validate())
        widget.grid(row=row, column=1, sticky="w", pady=2)

        # La ayuda del parámetro que se toca, sin ir a buscarla.
        for target in (label, widget):
            target.bind("<Enter>", lambda _e, p=parameter: self._show_help(
                f"{p.label}\n\n{p.help}"))
            target.bind("<FocusIn>", lambda _e, p=parameter: self._show_help(
                f"{p.label}\n\n{p.help}"))
        self._vars[parameter.name] = variable
        self._rows.append((parameter, variable))

    # --------------------------------------------------------------- valores

    def values(self) -> dict[str, Any]:
        """Lo tecleado, convertido a lo que el catálogo espera.

        Lo que no se pueda convertir se deja crudo a propósito: la
        validación lo nombrará con su mensaje, que dice más que un
        ``ValueError`` del widget.
        """
        chosen = kind(self.current_kind)
        out = chosen.defaults()
        for parameter, variable in self._rows:
            raw = variable.get()
            out[parameter.name] = _coerce(parameter, raw)
        return out

    # ------------------------------------------------------------ validación

    def problems(self) -> list[str]:
        """Lo que impide añadir este paso, ahora mismo."""
        return kind(self.current_kind).check(self.values(), self.atoms)

    def _validate(self) -> None:
        problems = list(self.problems())
        recipe = self.recipe()
        if self.steps:
            problems.extend(f"(receta) {p}" for p in recipe.check(self.atoms))
        self.txt_problems.configure(state="normal")
        self.txt_problems.delete("1.0", "end")
        self.txt_problems.insert(
            "1.0", "\n".join(f"• {p}" for p in problems)
            or "Sin problemas con estos valores.")
        self.txt_problems.configure(state="disabled")
        if self.on_change is not None:
            self.on_change(problems)

    def _show_help(self, text: str) -> None:
        self.txt_help.configure(state="normal")
        self.txt_help.delete("1.0", "end")
        self.txt_help.insert("1.0", text)
        self.txt_help.configure(state="disabled")

    # ---------------------------------------------------------------- pasos

    def add_step(self) -> bool:
        """Añade el cálculo actual a la receta, si se puede."""
        blocking = [p for p in self.problems() if not p.startswith("Nota")]
        if blocking:
            self._validate()
            return False
        self.steps.append(RecipeStep(self.current_kind, self.values()))
        self._refresh_steps()
        self._validate()
        return True

    def remove_last(self) -> None:
        if self.steps:
            self.steps.pop()
            self._refresh_steps()
            self._validate()

    def clear(self) -> None:
        self.steps.clear()
        self._refresh_steps()
        self._validate()

    def recipe(self) -> Recipe:
        """La receta compuesta hasta ahora."""
        return Recipe(name=self.var_name.get() or "calculo",
                      steps=list(self.steps))

    def _refresh_steps(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for position, step in enumerate(self.steps, start=1):
            self.tree.insert("", "end", iid=str(position),
                             text=f"{position}. {KINDS[step.kind].label}")


def _coerce(parameter: Parameter, raw: Any) -> Any:
    """Lo tecleado al tipo del catálogo; crudo si no convierte."""
    if parameter.type == "bool":
        return bool(raw)
    if parameter.type == "indices":
        text = str(raw).replace(",", " ").split()
        try:
            return [int(piece) for piece in text]
        except ValueError:
            return raw
    if parameter.type in ("int", "float"):
        try:
            # Un usuario en español teclea 0,05 tan a menudo como 0.05.
            number = float(str(raw).replace(",", "."))
        except (TypeError, ValueError):
            return raw
        return int(number) if parameter.type == "int" and number.is_integer() \
            else number
    return raw
