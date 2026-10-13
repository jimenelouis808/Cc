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
#: An axis declared periodic whose largest atom-free slab is at least this wide (Å) is
#: vacuum, not a lattice direction: a tube or a sheet in a box. Below it the declaration
#: stands -- graphite's 3.35 Å interlayer gap and a molecular crystal's contacts are real
#: periodicity that no covalent bond crosses.
VACUUM_GAP = 5.0


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
        if key == "CELL" and len(line.split()) >= 7:    # Materials Studio: values inline
            cell = [float(x) for x in line.split()[1:7]]
            section = None
            continue
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
    return crossing_axes(atoms, factor)


def crossing_axes(atoms: Atoms, factor: float = BOND_FACTOR) -> list[bool]:
    """Axes along which bonds cross the cell, whatever ``pbc`` declares (needs a cell)."""
    if atoms.cell.rank < 3 or len(atoms) < 2:
        return [False] * 3
    from .neighbors import neighbor_list

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


def vacuum_gaps(atoms: Atoms) -> list[float | None]:
    """Per cell axis, the widest slab with no atom in it (Å, measured perpendicular to the
    other two vectors, across the boundary too). ``None`` without a cell."""
    if atoms.cell.rank < 3:
        return [None] * 3
    cell = np.asarray(atoms.cell)
    volume = abs(np.linalg.det(cell))
    fractions = np.mod(atoms.get_scaled_positions(wrap=False), 1.0)
    out = []
    for axis in range(3):
        others = [cell[k] for k in range(3) if k != axis]
        spacing = volume / np.linalg.norm(np.cross(*others))     # distance between planes
        f = np.sort(fractions[:, axis])
        steps = np.diff(np.append(f, f[0] + 1.0)) if len(f) else np.array([1.0])
        out.append(float(steps.max() * spacing))
    return out


def auto_periodicity(atoms: Atoms, declared=None) -> tuple[list[bool], list[str]]:
    """The periodic axes a structure has, and why, axis by axis.

    Periodic when bonds cross the cell along it. An axis no bond crosses keeps what the
    file ``declared`` (default: the current ``pbc``) unless it is vacuum -- a free slab of
    :data:`VACUUM_GAP` or more -- since a box around a tube or a sheet is not a lattice.
    No cell, no periodic axis: a cell is never invented.
    """
    declared = [bool(p) for p in (atoms.get_pbc() if declared is None else declared)]
    if atoms.cell.rank < 3:
        return [False] * 3, ["sin celda"] * 3
    crossing = crossing_axes(atoms)
    gaps = vacuum_gaps(atoms)
    axes, reasons = [], []
    for axis in range(3):
        if crossing[axis]:
            axes.append(True)
            reasons.append("enlaces que cruzan la celda")
        elif gaps[axis] >= VACUUM_GAP:
            axes.append(False)
            reasons.append(f"{gaps[axis]:.1f} Å de vacío"
                           + (" (el archivo lo declaraba periódico)" if declared[axis] else ""))
        elif declared[axis]:
            axes.append(True)
            reasons.append(f"declarado periódico; sin enlaces que crucen, {gaps[axis]:.1f} Å "
                           "entre imágenes")
        else:
            axes.append(False)
            reasons.append(f"declarado no periódico; solo {gaps[axis]:.1f} Å hasta la imagen")
    return axes, reasons


def dimensionality(pbc) -> str:
    """'0D (finita)', '1D (c)', '2D (a, b)', '3D'."""
    axes = [k for k in range(3) if pbc[k]]
    if not axes:
        return "0D (finita)"
    if len(axes) == 3:
        return "3D"
    return f"{len(axes)}D ({', '.join('abc'[k] for k in axes)})"


def align_wire(atoms: Atoms, vacuum: float = 7.0) -> Atoms:
    """A 1D structure (one periodic axis) turned so the periodic vector lies on z, in an
    orthorhombic box with ``vacuum`` Å on each open side; a finite one boxed the same way.

    The open axes then are x and y, perpendicular to the lattice vector: what a field
    across the wire needs (:func:`tbkit.infrared.born_by_field`), and what a code with zero
    boundary conditions along open axes (GPAW) needs. A tilted cell -- the user's helix came
    with its c axis 0.5° off z -- is otherwise refused there. Bonds are checked to be the
    same before and after. A structure periodic in 2 or 3 axes is returned unchanged.
    """
    pbc = [bool(p) for p in atoms.get_pbc()]
    if sum(pbc) > 1:
        return atoms.copy()
    from .neighbors import neighbor_list

    out = atoms.copy()
    before = len(neighbor_list("i", out, 1.85)) if len(out) > 1 else 0
    if sum(pbc) == 1:
        axis = pbc.index(True)
        vector = np.array(out.cell[axis])
        length = float(np.linalg.norm(vector))
        out.rotate(vector, (0.0, 0.0, 1.0), rotate_cell=True)
        positions = out.get_positions()
        span = positions[:, :2].max(axis=0) - positions[:, :2].min(axis=0)
        out.set_cell(np.diag([span[0] + 2 * vacuum, span[1] + 2 * vacuum, length]),
                     scale_atoms=False)
        out.pbc = [False, False, True]
        out.translate([vacuum - positions[:, 0].min(), vacuum - positions[:, 1].min(), 0.0])
        out.wrap()
    else:
        out.pbc = False
        out.center(vacuum=vacuum)
    after = len(neighbor_list("i", out, 1.85)) if len(out) > 1 else 0
    if after != before:
        raise ValueError(f"Al alinear cambió el número de enlaces ({before // 2} → "
                         f"{after // 2}): revisa la celda y la periodicidad.")
    for key in ("periodicity", "periodicity_mode", "pbc_file"):
        out.info.pop(key, None)
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
    if guess_periodic and atoms.cell.rank == 3:
        declared = [bool(p) for p in atoms.get_pbc()]
        axes, reasons = auto_periodicity(atoms, declared)
        atoms.info["pbc_file"] = "".join("TF"[not p] for p in declared)
        atoms.info["periodicity_mode"] = "auto"
        if axes != declared:
            atoms.pbc = axes
            atoms.info["periodicity"] = (
                f"periodicidad automática: {dimensionality(axes)} (el archivo declaraba "
                f"{dimensionality(declared)}) · "
                + "; ".join(f"{'abc'[k]}: {r}" for k, r in enumerate(reasons)))
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
