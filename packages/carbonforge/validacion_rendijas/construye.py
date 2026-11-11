"""Build a series of graphene slit pores, one per wall separation.

The point of this series is that it is the one geometry where the pore
width can be moved while everything else is held fixed: the chemistry is
graphene's, the curvature is zero, and the wall is one atom thick --
which is what the superstructures' walls are too, so this is their
control and not a different material.

`H` is the centre-to-centre distance between adjacent carbon planes. The
width a molecule actually sees is smaller by roughly a carbon diameter;
`analyse_porosity` measures that directly and is the number to quote.

Three things are forced by the GCMC rather than chosen:

* every cell edge must be at least twice the 12 A cutoff, so the sheet is
  tiled to ~25 A in x and y and the stack is repeated along z until it
  clears 24 A as well. Repeating a periodic axis is not a modelling
  decision -- it is the same crystal -- which is what makes a 6 A slit
  measurable at all.
* the cell must be orthorhombic, so the rectangular supercell is used
  rather than the hexagonal primitive one.
* stacking is AA. At these separations the registry between layers is
  invisible to a hydrogen molecule, and AA keeps every gap identical,
  which is the whole point of the series.

The sheet comes from carbonforge's OWN graphene builder, not from the
sibling package's. The two are byte-identical here -- same cell, same
positions, checked with `np.array_equal`, which they would be, having a
common ancestor -- but the packages do not import each other, so a
validation folder inside carbonforge uses carbonforge's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase.io import write

from carbonforge.builders.graphene import build_graphene_supercell

CUTOFF = 12.0
MIN_EDGE = 2.0 * CUTOFF
SEPARATIONS = (6.0, 7.0, 8.0, 10.0, 14.0, 20.0)


def slit(separation: float, out: Path) -> dict:
    """One slit pore: AA-stacked graphene planes `separation` apart."""
    sheet = build_graphene_supercell(nx=10, ny=6, vacuum=10.0)
    sheet.set_cell(np.diag([sheet.cell[0, 0], sheet.cell[1, 1], separation]))
    sheet.center(axis=2)
    layers = max(1, int(np.ceil(MIN_EDGE / separation)))
    stack = sheet * (1, 1, layers)
    stack.set_pbc((True, True, True))
    stack.wrap()
    edges = np.diag(stack.cell)
    if min(edges) < MIN_EDGE:
        raise ValueError(f"edge {min(edges):.1f} A under the {MIN_EDGE:.0f} A "
                         "the minimum image needs")
    write(str(out), stack, format="extxyz")
    return {"separacion_a": separation, "capas": layers,
            "atomos": len(stack), "celda": [round(float(e), 2) for e in edges]}


def main() -> int:
    """Write one file per separation and report what was built.

    Separations may be given after the folder; without them the default
    series is built.
    """
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    root.mkdir(parents=True, exist_ok=True)
    wanted = [float(a) for a in sys.argv[2:]] or list(SEPARATIONS)
    for separation in wanted:
        info = slit(separation, root / f"rendija_{separation:g}.extxyz")
        print(f"H {info['separacion_a']:5.1f} A: {info['capas']} capas, "
              f"{info['atomos']:5d} atomos, celda {info['celda']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
