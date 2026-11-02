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


def within_bonds(atoms: Atoms, bond_cut: float, steps: int) -> set:
    """``(i, j, shift)`` for every pair joined by a path of at most ``steps`` bonds (a bond:
    closer than ``bond_cut``), ``i`` itself included with shift 0. Lattice shifts follow
    the path, so an image of ``i`` reached through the bonds counts as joined."""
    ii, jj, ss = neighbor_list("ijS", atoms, bond_cut)
    graph = [[] for _ in range(len(atoms))]
    for i, j, sh in zip(ii, jj, ss, strict=True):
        graph[i].append((int(j), (int(sh[0]), int(sh[1]), int(sh[2]))))
    out = set()
    for i in range(len(atoms)):
        front = {(i, (0, 0, 0))}
        seen = set(front)
        for _ in range(steps):
            nxt = set()
            for a, sa in front:
                for b, sb in graph[a]:
                    key = (b, (sa[0] + sb[0], sa[1] + sb[1], sa[2] + sb[2]))
                    if key not in seen:
                        seen.add(key)
                        nxt.add(key)
            front = nxt
        out.update((i, b, sb) for b, sb in seen)
    return out


def far_mask(atoms: Atoms, ii, jj, shifts, bond_cut: float, steps: int) -> np.ndarray:
    """True for the pairs of a neighbour list that are more than ``steps`` bonds apart."""
    near = within_bonds(atoms, bond_cut, steps)
    return np.array([(int(i), int(j), (int(s[0]), int(s[1]), int(s[2]))) not in near
                     for i, j, s in zip(ii, jj, shifts, strict=True)], dtype=bool)
