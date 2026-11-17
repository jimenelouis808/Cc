"""Is a cage's uptake the cage, or the gap between cages?

A cage is finite in all three directions, so the handoff closes each
axis with a gap between images -- 3.4 A, the graphitic contact. That
choice puts the outer walls of neighbouring cages 3.4 A apart, which
is a SLIT, and close to the optimal one: the best slit for H2 at 77 K
has a 3.55 A free channel. So a high coefficient may be measuring the
interstices of a packing rather than the cage.

The control is to open the gap. If the uptake is in the cage, widening
the gap only adds dead volume and the coefficient per gram barely
moves. If it is in the interstices, the coefficient collapses.

Measured on the two cages that pass the channel triage, the answer is
that the cage holds it: opening the gap from 3.4 to 14 Å moves `K_H`
by 15-20 % and `q_st` not at all.

    super-cube          3.4 -> 51655   8.0 -> 41481   14.0 -> 44520
    super-tetrahedron          33221          27639          31181

So the packing choice, which the handoff has to make and which a
coefficient per gram does depend on, is not what produces the number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

from carbonforge.adsorption.henry import henry_constant
from carbonforge.adsorption.porosity import analyse_porosity

#: Twice the 12 Å cutoff, as everywhere in this folder.
MIN_EDGE = 24.0

#: Gaps to compare, in Å. 3.4 is the graphitic contact the handoff
#: uses; 14 leaves no slit between images at all.
GAPS = (3.4, 8.0, 14.0)

#: Insertions per seed, and two seeds.
SAMPLES = 400_000
SEEDS = (0, 1)


def embed_with_gap(atoms: Atoms, gap: float) -> Atoms:
    """Close every free axis with `gap`, then replicate to the minimum."""
    cell = np.asarray(atoms.cell, dtype=float)
    edges = np.diag(cell).astype(float).copy()
    positions = atoms.get_positions()
    for axis in range(3):
        if atoms.pbc[axis]:
            continue
        span = positions[:, axis].max() - positions[:, axis].min()
        edges[axis] = span + gap
        positions[:, axis] -= positions[:, axis].min() - gap / 2.0
    out = atoms.copy()
    out.set_positions(positions)
    out.set_cell(np.diag(edges))
    out.set_pbc(True)
    out.wrap()
    repeat = [max(1, int(np.ceil(MIN_EDGE / e))) for e in edges]
    return out * tuple(repeat)


if len(sys.argv) < 3:
    print(__doc__)
    raise SystemExit(2)
folder = Path(sys.argv[1])
print("| estructura | hueco (Å) | átomos | canal (Å) | K_H | q_st |")
print("|---|---|---|---|---|---|")
for name in sys.argv[2:]:
    atoms = read(str(folder / f"{name}.extxyz"))
    for gap in GAPS:
        big = embed_with_gap(atoms, gap)
        runs = [henry_constant(big, temperature=77.0, samples=SAMPLES,
                               seed=seed) for seed in SEEDS]
        v = [r.henry_coefficient for r in runs]
        por = analyse_porosity(big, samples=20_000, area_points=200, seed=0)
        print(f"| {name} | {gap:.1f} | {len(big)} | "
              f"{por.largest_free_sphere:.2f} | {np.mean(v):.0f} ± "
              f"{0.5*(max(v)-min(v)):.0f} | {runs[0].isosteric_heat:.2f} |",
              flush=True)
