"""El panel que enseña la estructura y deja elegir átomos.

Delgado a propósito: todo lo que piensa está en
:mod:`carbonforge.dft.structure`, que se prueba sin Tk. Aquí sólo hay
widgets, y la regla que ya costó cara en otras secciones de la suite --
dentro de un padre, lo que se empaqueta DESPUÉS de algo con
``expand=True`` se queda sin espacio-- se respeta poniendo primero lo que
no debe desaparecer.

Tres zonas:

* **la caja**: ejes periódicos, vacío por lado y el veredicto para el modo
  elegido, que es donde se ve antes de lanzar que un vacío que sobra en
  LCAO se queda corto en ondas planas;
* **la tabla de átomos**, con índice, símbolo, coordinación, vecinos y
  anillos, ordenable y filtrable;
* **los entornos**, que es por donde se elige de verdad: un clic en "N
  piridínico" selecciona su representante, no los catorce equivalentes.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable, Optional

from ase import Atoms

from .structure import StructureView, describe, representatives, select

__all__ = ["StructurePanel"]


class StructurePanel(ttk.Frame):
    """Caja, átomos y entornos de una estructura, con selección."""

    def __init__(self, master, *, on_selection: Optional[Callable] = None,
                 **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.on_selection = on_selection
        self.view: Optional[StructureView] = None
        self._selected: set[int] = set()
        self.var_mode = tk.StringVar(value="lcao")
        self.var_filter = tk.StringVar(value="")

        # La cabecera y los botones van antes que las tablas, que son las
        # que se expanden: al revés se quedarían en cero píxeles.
        header = ttk.Frame(self)
        header.pack(fill="x", padx=6, pady=(6, 0))
        self.lbl_formula = ttk.Label(header, text="(sin estructura)",
                                     font=("TkDefaultFont", 10, "bold"))
        self.lbl_formula.pack(side="left")
        ttk.Label(header, text="  modo:").pack(side="left")
        ttk.Combobox(header, textvariable=self.var_mode, width=6,
                     values=("lcao", "fd", "pw"), state="readonly"
                     ).pack(side="left")
        self.var_mode.trace_add("write", lambda *_: self._refresh_cell())

        self.txt_cell = tk.Text(self, height=7, wrap="word", relief="flat",
                                font=("TkFixedFont", 8))
        self.txt_cell.pack(fill="x", padx=6, pady=(4, 0))
        self.txt_cell.configure(state="disabled")

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=6, pady=4)
        ttk.Button(buttons, text="Un átomo por entorno",
                   command=self.select_representatives).pack(side="left")
        ttk.Button(buttons, text="Limpiar selección",
                   command=self.clear_selection).pack(side="left", padx=4)
        ttk.Label(buttons, text="filtrar:").pack(side="left", padx=(8, 2))
        entry = ttk.Entry(buttons, textvariable=self.var_filter, width=12)
        entry.pack(side="left")
        self.var_filter.trace_add("write", lambda *_: self._fill_atoms())

        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        left = ttk.Labelframe(panes, text="Entornos", padding=4)
        self.tree_env = ttk.Treeview(
            left, columns=("n",), show="tree headings", height=8,
            selectmode="browse")
        self.tree_env.heading("#0", text="entorno")
        self.tree_env.heading("n", text="n")
        self.tree_env.column("n", width=40, anchor="e")
        self.tree_env.pack(fill="both", expand=True)
        self.tree_env.bind("<<TreeviewSelect>>", self._on_environment)
        panes.add(left, weight=1)

        right = ttk.Labelframe(panes, text="Átomos", padding=4)
        columns = ("sym", "coord", "vecinos", "anillos", "entorno")
        self.tree_atoms = ttk.Treeview(
            right, columns=columns, show="tree headings", height=14,
            selectmode="extended")
        self.tree_atoms.heading("#0", text="i")
        self.tree_atoms.column("#0", width=48, anchor="e")
        for name, width in zip(columns, (46, 46, 90, 70, 180)):
            self.tree_atoms.heading(name, text=name)
            self.tree_atoms.column(name, width=width, anchor="w")
        self.tree_atoms.pack(side="left", fill="both", expand=True)
        bar = ttk.Scrollbar(right, orient="vertical",
                            command=self.tree_atoms.yview)
        bar.pack(side="right", fill="y")
        self.tree_atoms.configure(yscrollcommand=bar.set)
        self.tree_atoms.bind("<<TreeviewSelect>>", self._on_atoms)
        panes.add(right, weight=3)

        self._refresh_cell()

    # ------------------------------------------------------------- cargar

    def show(self, atoms: Atoms) -> StructureView:
        """Analiza ``atoms`` y pinta las tres zonas."""
        self.view = describe(atoms)
        self._selected.clear()
        self.lbl_formula.config(text=self.view.formula)
        self._refresh_cell()
        self._fill_environments()
        self._fill_atoms()
        return self.view

    # -------------------------------------------------------------- caja

    def _refresh_cell(self) -> None:
        self.txt_cell.configure(state="normal")
        self.txt_cell.delete("1.0", "end")
        if self.view is None:
            self.txt_cell.insert("1.0", "Carga una estructura.")
            self.txt_cell.configure(state="disabled")
            return
        cell = self.view.cell
        axes = ", ".join(
            "{}={:.2f} Å ({})".format(
                "xyz"[i], cell.lengths[i],
                "periódico" if cell.pbc[i]
                else f"vacío {cell.vacuum_per_side[i]:.1f} Å/lado")
            for i in range(3))
        angles = (f"Ángulos: {cell.angles[0]:.1f}, {cell.angles[1]:.1f}, "
                  f"{cell.angles[2]:.1f}°   |   dimensionalidad: "
                  f"{cell.dimensionality}D")
        lines = [f"Celda: {axes}", angles]
        for axis in cell.periodic_axes:
            lines.append(
                f"Separación entre imágenes en {'xyz'[axis]}: "
                f"{cell.image_separation(axis):.2f} Å")
        problems = cell.vacuum_verdict(self.var_mode.get())
        lines.extend(problems or
                     [f"Vacío suficiente para {self.var_mode.get()}."])
        lines.extend(self.view.notes)
        self.txt_cell.insert("1.0", "\n".join(lines))
        self.txt_cell.configure(state="disabled")

    # ---------------------------------------------------------- entornos

    def _fill_environments(self) -> None:
        self.tree_env.delete(*self.tree_env.get_children())
        if self.view is None:
            return
        for environment in self.view.environments:
            self.tree_env.insert(
                "", "end", iid=environment.key, text=environment.label,
                values=(environment.count,))

    def _on_environment(self, _event=None) -> None:
        if self.view is None:
            return
        chosen = self.tree_env.selection()
        if not chosen:
            return
        for environment in self.view.environments:
            if environment.key == chosen[0]:
                # El representante, no los equivalentes: dan el mismo número.
                self.set_selection([environment.representative])
                return

    # ------------------------------------------------------------ átomos

    def _fill_atoms(self) -> None:
        self.tree_atoms.delete(*self.tree_atoms.get_children())
        if self.view is None:
            return
        needle = self.var_filter.get().strip()
        symbols = {row.symbol for row in self.view.rows}
        # Un simbolo quimico suelto significa la especie, no "donde aparezca
        # esa letra": escribir H y que salgan los carbonos "C sp2 con H" es
        # exactamente lo contrario de lo que se pedia.
        by_symbol = needle in symbols
        lowered = needle.lower()
        for row in self.view.rows:
            if needle:
                if by_symbol:
                    if row.symbol != needle:
                        continue
                elif lowered not in f"{row.index} {row.environment}".lower():
                    continue
            self.tree_atoms.insert(
                "", "end", iid=str(row.index), text=str(row.index),
                values=(row.symbol, row.coordination,
                        ",".join(str(n) for n in row.neighbours) or "—",
                        "/".join(str(r) for r in row.rings) or "—",
                        row.environment))
        self._apply_selection_to_tree()

    def _on_atoms(self, _event=None) -> None:
        chosen = {int(i) for i in self.tree_atoms.selection()}
        if chosen != self._selected:
            self._selected = chosen
            self._announce()

    # --------------------------------------------------------- selección

    @property
    def selection(self) -> tuple[int, ...]:
        """Los átomos elegidos, en orden."""
        return tuple(sorted(self._selected))

    def set_selection(self, indices: Iterable[int]) -> None:
        self._selected = {int(i) for i in indices}
        self._apply_selection_to_tree()
        self._announce()

    def clear_selection(self) -> None:
        self.set_selection(())

    def select_representatives(self, only: Optional[Iterable[str]] = None) -> None:
        """Un átomo por entorno: la lista que un XPS necesita calcular."""
        if self.view is not None:
            self.set_selection(representatives(self.view, only=only))

    def select_where(self, **criteria) -> None:
        """Selecciona por los criterios de :func:`~.structure.select`."""
        if self.view is not None:
            self.set_selection(select(self.view, **criteria))

    def _apply_selection_to_tree(self) -> None:
        visible = set(self.tree_atoms.get_children())
        wanted = [str(i) for i in sorted(self._selected) if str(i) in visible]
        self.tree_atoms.selection_set(wanted)
        if wanted:
            self.tree_atoms.see(wanted[0])

    def _announce(self) -> None:
        if self.on_selection is not None:
            self.on_selection(self.selection)
