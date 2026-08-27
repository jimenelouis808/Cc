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
    """Return the per-atom coordination number as a NumPy integer array."""
    g = build_bond_graph(atoms, tolerance=tolerance)
    n = len(atoms)
    coord = np.zeros(n, dtype=int)
    for i in range(n):
        coord[i] = g.degree[i] if i in g else 0
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
