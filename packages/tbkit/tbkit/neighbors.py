"""``ase.neighborlist.neighbor_list`` with a fast path for cells that are not periodic
along every axis.

ASE's search is hundreds of times slower when an axis with a cell vector is not periodic:
on a 480-atom helix in a 60 × 60 × 16 Å box, periodic in z only, one call took 12.7 s
against 0.03 s with every axis periodic -- 72 % of a force evaluation, and a
finite-difference Hessian makes thousands of them. So the search runs with every axis
periodic and the pairs that cross a non-periodic axis are dropped afterwards. That is exact
whatever the vacuum: a genuine pair has no lattice shift along a non-periodic axis (its
vector is the plain difference of positions there), and every pair with such a shift is an
image the real structure does not have.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list as _ase_neighbor_list


def neighbor_list(quantities: str, atoms: Atoms, cutoff, self_interaction: bool = False,
                  max_nbins: float = 1e6):
    """Same call and results as ASE's ``neighbor_list`` (same order of pairs not promised).

    The fast path applies when the cell has three vectors and some axis is not periodic.
    """
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    if pbc.all() or atoms.cell.rank < 3:
        return _ase_neighbor_list(quantities, atoms, cutoff, self_interaction=self_interaction,
                                  max_nbins=max_nbins)
    probe = atoms.copy()
    probe.calc = None
    probe.pbc = True
    asked = quantities if "S" in quantities else quantities + "S"
    out = _ase_neighbor_list(asked, probe, cutoff, self_interaction=self_interaction,
                             max_nbins=max_nbins)
    shifts = out[asked.index("S")]
    keep = ~np.any(shifts[:, ~pbc] != 0, axis=1)
    result = tuple(np.asarray(a)[keep] for a in out[:len(quantities)])
    return result[0] if len(quantities) == 1 else result
