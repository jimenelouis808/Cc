"""networkx-based topology utilities.

The bond graph abstracts the structure as an undirected graph of atoms
connected by covalent bonds (cutoff inferred from covalent radii). On top
of the graph we compute coordination, connectivity and small-ring
statistics, which are standard descriptors for nanocarbon topology.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
from ase import Atoms

from ..utils.geometry import guess_bonds


def build_bond_graph(atoms: Atoms, tolerance: float = 0.30) -> nx.Graph:
    """Build an undirected bond graph of the structure.

    Nodes are atom indices (with ``element`` and ``position`` attributes).
    Edges carry the bond ``distance`` in Å.

    Parameters
    ----------
    atoms
        Structure to analyse.
    tolerance
        Extra slack (Å) added to the sum of covalent radii when guessing bonds.
    """
    g = nx.Graph()
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    for i, sym in enumerate(symbols):
        g.add_node(i, element=sym, position=tuple(positions[i]))
    for i, j, d in guess_bonds(atoms, tolerance=tolerance):
        g.add_edge(i, j, distance=d)
    return g


def coordination_numbers(atoms: Atoms, tolerance: float = 0.30) -> np.ndarray:
    """Return the per-atom coordination number as a NumPy integer array.

    Uses ``atoms.info["bonds"]`` when the builder recorded one, and falls
    back to distances otherwise. That preference matters on a curved
    structure: a 2.4 Å M-X bond's distance cutoff reaches ~2.9 Å, and on
    a minimal surface plenty of *non*-bonded atoms sit inside that, so a
    schwarzite whose every metal has exactly six bonds reads as
    ten-coordinate and fails validation. The same reasoning already
    forbids re-deriving rings from distances on a curved shell.
    """
    n = len(atoms)
    recorded = atoms.info.get("bonds")
    if recorded is not None and len(recorded):
        pairs = np.asarray(recorded, dtype=int)
        if pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.max() < n:
            coord = np.zeros(n, dtype=int)
            np.add.at(coord, pairs[:, 0], 1)
            np.add.at(coord, pairs[:, 1], 1)
            return coord

    # Counted from the bond list, not from the graph's node degrees. A
    # networkx Graph holds one edge per *pair*, and in a cell only one or
    # two repeats wide an atom reaches the same neighbour through several
    # images -- so the degree undercounts. A 1x1 MoS2 cell has 12 M-X
    # bonds and its metals are six-coordinate, but the collapsed graph
    # gives each of them 2, and validation then warned about "dangling"
    # chalcogens in a perfect crystal.
    coord = np.zeros(n, dtype=int)
    for first, second, _ in guess_bonds(atoms, tolerance=tolerance):
        coord[first] += 1
        coord[second] += 1
    return coord


def connected_components(
    atoms: Atoms, tolerance: float = 0.30
) -> list[list[int]]:
    """Return the list of connected-component atom-index lists, largest first."""
    g = build_bond_graph(atoms, tolerance=tolerance)
    comps = [sorted(list(c)) for c in nx.connected_components(g)]
    comps.sort(key=len, reverse=True)
    return comps


def ring_statistics(
    atoms: Atoms,
    max_ring: int = 8,
    tolerance: float = 0.30,
) -> dict[int, int]:
    """Count shortest rings up to size ``max_ring``.

    Uses networkx's ``cycle_basis`` on the bond graph: this returns a minimum
    cycle basis, which for sp2 carbon corresponds to the expected face rings
    in flat systems (hexagons in graphene, pentagons/heptagons around
    Stone-Wales or topological defects). On 3D disordered foams it is only
    indicative, since cycle basis is not unique.

    Parameters
    ----------
    atoms
        Structure to analyse.
    max_ring
        Largest ring size to report.
    tolerance
        Bond-detection tolerance.

    Returns
    -------
    dict
        Mapping ``ring_size → count`` for sizes in ``[3, max_ring]``.
    """
    g = build_bond_graph(atoms, tolerance=tolerance)
    counts: Counter[int] = Counter()
    for cycle in nx.cycle_basis(g):
        size = len(cycle)
        if 3 <= size <= max_ring:
            counts[size] += 1
    return {k: counts.get(k, 0) for k in range(3, max_ring + 1)}
