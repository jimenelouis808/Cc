"""Embed a super-graphene sheet in a cell the GCMC will accept.

Super-graphene is periodic in x and y and finite in z, and the GCMC
needs three periodic axes, an orthorhombic cell and every edge at least
twice the 12 A cutoff. Two different things get it there:

* the periodic axes are REPLICATED until they clear 24 A. That decides
  nothing -- a replicated cell is the same crystal -- and it is checked
  by counting the bonds under the periodic boundary against the bond
  list the builder itself wrote.
* the free axis gets a GAP between images. That is a packing choice, so
  it is an argument, it is recorded, and the same 3.4 A (the graphitic
  contact) is used across the series so the widths stay comparable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write
from scipy.spatial import cKDTree

CUTOFF = 12.0
MIN_EDGE = 2.0 * CUTOFF


def periodic_bond_count(atoms, lo: float = 1.1, hi: float = 1.8) -> int:
    """Bonds under the cell's own periodic boundary."""
    tree = cKDTree(atoms.get_positions(), boxsize=np.diag(atoms.cell))
    pairs = tree.query_pairs(hi, output_type="ndarray")
    if len(pairs) == 0:
        return 0
    d = np.linalg.norm(
        atoms.get_distances(pairs[:, 0], pairs[:, 1], mic=True, vector=True),
        axis=1)
    return int(((d > lo) & (d < hi)).sum())


def embed(source: Path, gap: float, out: Path) -> dict:
    """Orthorhombic, 3-periodic, every edge >= 2*cutoff.

    The cell comes from the builder's own json, not from the .xyz: the
    builder writes a plain XYZ, which carries no lattice at all, and ASE
    reads it as a zero cell. Silently embedding into that would have
    produced a structure with no periodicity and a porosity report about
    nothing.
    """
    atoms = read(str(source))
    record = json.loads(source.with_suffix(".json").read_text())
    lattice = np.array(record["cell"], dtype=float)
    off = np.abs(lattice - np.diag(np.diag(lattice))).max()
    if off > 1e-9:
        raise ValueError(f"{source.name}: the cell is not orthorhombic "
                         f"(largest off-diagonal {off:.3g} A), and every "
                         "neighbour search downstream is orthorhombic-only")
    atoms.set_cell(lattice)
    positions = atoms.get_positions()
    cell = np.diag(lattice).astype(float).copy()
    # z is the free axis: centre the sheet and give it the gap.
    span = positions[:, 2].max() - positions[:, 2].min()
    cell[2] = span + gap
    positions[:, 2] -= positions[:, 2].min() - gap / 2.0
    atoms.set_positions(positions)
    atoms.set_cell(np.diag(cell))
    atoms.set_pbc((True, True, True))
    atoms.wrap()
    repeat = [max(1, int(np.ceil(MIN_EDGE / e))) for e in cell]
    big = atoms * tuple(repeat)
    write(str(out), big, format="extxyz")
    return {"fuente": source.name, "separacion_a": gap,
            "repeticion": repeat, "atomos": len(big),
            "celda": [round(float(e), 2) for e in np.diag(big.cell)],
            "enlaces_pbc": periodic_bond_count(big)}


def main() -> int:
    """Embed every .xyz given, at the gap given first."""
    gap = float(sys.argv[1])
    for name in sys.argv[2:]:
        source = Path(name)
        info = embed(source, gap, source.with_suffix(".extxyz"))
        record = json.loads(source.with_suffix(".json").read_text())
        expected = len(record.get("bonds", [])) * int(
            np.prod(info["repeticion"]))
        info["enlaces_del_constructor"] = expected
        ok = "=" if info["enlaces_pbc"] == expected else "DISTINTO DE"
        print(f"{source.stem}: {info['atomos']:5d} átomos, celda "
              f"{info['celda']}, repetición {info['repeticion']}, "
              f"{info['enlaces_pbc']} enlaces bajo PBC {ok} "
              f"{expected} del constructor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
