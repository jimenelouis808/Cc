"""Reading structures: every format ASE knows, plus Materials Studio ``.xtl``, and a guess
of which directions are periodic when the file does not say.

An ``.xyz`` has no standard place for the cell: plain XYZ has none, and an extended XYZ may
carry ``Lattice=`` with ``pbc="F F F"`` (a molecule in a box) or no ``pbc`` at all. Only the
file can say what the structure is, so this module never *invents* a cell; it does the one
thing that can be checked, for a structure that has a cell but is declared non-periodic:
with every axis closed, an axis is periodic when opening it loses bonds (atoms of one face
bonded to the opposite face). A molecule in a box has none.
:func:`with_cell` sets a cell and the periodic axes by hand (the window's «Celda…»).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

#: Bonded when closer than this times the sum of covalent radii (as the viewer draws them).
BOND_FACTOR = 1.2


def _element(label: str) -> str:
    """The chemical symbol of an atom label (``C1``, ``Ca2``, ``N_a``, ``CL``)."""
    letters = re.match(r"[A-Za-z]+", label.strip())
    if letters is None:
        raise ValueError(f"No reconozco el elemento de la etiqueta {label!r}.")
    word = letters.group(0)
    for size in (2, 1):
        symbol = word[:size].capitalize()
        # two letters only when the second is not the start of an index or suffix
        if len(symbol) == size and symbol in atomic_numbers and (
                size == 1 or word[1].islower() or len(word) == 2):
            return symbol
    raise ValueError(f"No reconozco el elemento de la etiqueta {label!r}.")


def read_xtl(path: str | Path) -> Atoms:
    """Materials Studio ``.xtl``: ``CELL`` (a b c α β γ), ``SYMMETRY NUMBER`` and fractional
    ``ATOMS``. Space groups other than P1 are expanded with ASE's tables (standard setting);
    the file lists no operations, so a non-standard setting cannot be recovered."""
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8",
                                                       errors="replace").splitlines()]
    cell = None
    number = 1
    labels: list[str] = []
    fractions: list[list[float]] = []
    section = None
    columns: list[str] = []
    for line in lines:
        if not line:
            continue
        key = line.split()[0].upper()
        if key == "EOF":
            break
        if key in ("TITLE", "DIMENSION", "SYMMETRY", "CELL", "ATOMS", "ATOM"):
            section = key if key != "ATOM" else "ATOMS"
            if key == "SYMMETRY" and line.upper().split()[1:2] == ["NUMBER"]:
                number = int(line.split()[2])
            if key == "DIMENSION" and int(line.split()[1]) != 3:
                raise ValueError("Solo se leen archivos .xtl de dimensión 3.")
            continue
        if section == "CELL":
            values = [float(x) for x in line.split()[:6]]
            if len(values) != 6:
                raise ValueError("La línea de CELL del .xtl debe traer a b c α β γ.")
            cell = values
            section = None
        elif section == "ATOMS":
            if key == "NAME":
                columns = line.upper().split()
                continue
            parts = line.split()
            x, y, z = (columns.index(c) for c in ("X", "Y", "Z")) if columns else (1, 2, 3)
            labels.append(parts[0])
            fractions.append([float(parts[x]), float(parts[y]), float(parts[z])])
    if cell is None or not labels:
        raise ValueError(f"{Path(path).name}: falta CELL o ATOMS; ¿es un .xtl?")
    symbols = [_element(label) for label in labels]
    if number > 1:
        from ase.spacegroup import crystal

        atoms = crystal(symbols, fractions, spacegroup=number, cellpar=cell)
    else:
        atoms = Atoms(symbols, scaled_positions=fractions, cell=cell, pbc=True)
    atoms.info["source_format"] = f"xtl (grupo {number})"
    return atoms


def periodic_axes(atoms: Atoms, factor: float = BOND_FACTOR) -> list[bool]:
    """Axes that close bonds across the cell boundary (see the module note). A structure with
    no cell, or one already periodic, returns its own ``pbc``."""
    if atoms.cell.rank < 3 or len(atoms) < 2:
        return [bool(p) for p in atoms.get_pbc()]
    from ase.neighborlist import neighbor_list

    def bonds(pbc):
        probe = atoms.copy()
        probe.pbc = pbc
        i, j, d = neighbor_list("ijd", probe, 3.5)
        limit = factor * (covalent_radii[probe.numbers[i]] + covalent_radii[probe.numbers[j]])
        return int(np.sum(d < limit))

    # Close every axis, then open one at a time: an axis is periodic when opening it loses
    # bonds (opening it alone is the test, because a bond can cross two faces at once).
    closed = bonds([True, True, True])
    out = []
    for axis in range(3):
        pbc = [True] * 3
        pbc[axis] = False
        out.append(bonds(pbc) < closed)
    return out


def read_structure(path: str | Path, guess_periodic: bool = True) -> Atoms:
    """``Atoms`` from a file. With ``guess_periodic``, a structure that has a cell but is
    declared non-periodic gets the axes :func:`periodic_axes` finds, and says so in
    ``atoms.info['periodicity']``."""
    path = Path(path)
    if path.suffix.lower() == ".xtl":
        atoms = read_xtl(path)
    else:
        from ase.io import read

        atoms = read(str(path))
    if isinstance(atoms, list):
        atoms = atoms[-1]
    if guess_periodic and not atoms.get_pbc().any() and atoms.cell.rank == 3:
        axes = periodic_axes(atoms)
        if any(axes):
            atoms.pbc = axes
            atoms.info["periodicity"] = (
                "periodicidad deducida de los enlaces que cruzan la celda: "
                + "".join("xyz"[k] for k, on in enumerate(axes) if on)
                + " (el archivo la declaraba no periódica)")
    return atoms


def read(path: str | Path, **_ignored) -> Atoms:
    """Drop-in for ``ase.io.read`` on structure files (``tbkit`` command line)."""
    return read_structure(path, guess_periodic=False)


def with_cell(atoms: Atoms, cell, pbc, scale_atoms: bool = False) -> Atoms:
    """A copy with a new cell and periodic axes. ``cell``: 3 lengths + 3 angles (6 numbers),
    3 lengths (orthorhombic), or a 3×3 matrix (Å); ``pbc``: three booleans. Atoms keep their
    Cartesian positions unless ``scale_atoms``."""
    values = np.asarray(cell, dtype=float)
    if values.shape == (3,):
        values = np.append(values, [90.0, 90.0, 90.0])
    if values.shape == (6,):
        from ase.cell import Cell

        values = Cell.fromcellpar(values).array
    if values.shape != (3, 3):
        raise ValueError("La celda son 3 longitudes, 3 longitudes y 3 ángulos, o una matriz 3×3.")
    out = atoms.copy()
    out.set_cell(values, scale_atoms=scale_atoms)
    out.pbc = [bool(p) for p in pbc]
    if out.pbc.any() and out.cell.rank < 3:
        raise ValueError("Un eje periódico necesita su vector de celda (no nulo).")
    return out


def write_xtl(atoms: Atoms, path: str | Path, title: str = "") -> Path:
    """Materials Studio ``.xtl`` in P1 (every atom listed, fractional coordinates)."""
    if atoms.cell.rank < 3:
        raise ValueError("Un .xtl necesita celda: fíjala antes («Celda…»).")
    lines = [f"TITLE {title or atoms.get_chemical_formula()}", "DIMENSION 3", "CELL",
             " " + " ".join(f"{v:.6f}" for v in atoms.cell.cellpar()),
             "SYMMETRY NUMBER 1", "SYMMETRY LABEL P1", "ATOMS", "NAME X Y Z"]
    counts: dict[str, int] = {}
    for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(),
                                 atoms.get_scaled_positions(wrap=False), strict=True):
        counts[symbol] = counts.get(symbol, 0) + 1
        lines.append(f"{symbol}{counts[symbol]} {x:.8f} {y:.8f} {z:.8f}")
    lines.append("EOF")
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_structure(atoms: Atoms, path: str | Path) -> Path:
    """Write by extension; ``.xyz`` as extended XYZ so the cell and the periodic axes
    survive (plain XYZ would drop them, the problem this module exists for)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xtl":
        return write_xtl(atoms, path)
    from ase.io import write

    if suffix in (".xyz", ".extxyz"):
        write(str(path), atoms, format="extxyz")
    elif path.name.upper().startswith(("POSCAR", "CONTCAR")) or suffix == ".vasp":
        write(str(path), atoms, format="vasp", direct=True)
    else:
        write(str(path), atoms)
    return path
