"""Small geometry helpers built on top of :mod:`ase`.

The helpers here are deliberately backend-agnostic: they operate on
:class:`ase.Atoms` objects and return plain NumPy arrays or new :class:`ase.Atoms`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import Atoms

from .constants import BOND_CUTOFF_OVERRIDE, COVALENT_RADII, MAX_CC_DISTANCE


def center_in_cell(atoms: Atoms, axes: Sequence[int] = (0, 1, 2)) -> Atoms:
    """Center atomic positions along the given cell axes.

    Parameters
    ----------
    atoms
        Input structure (modified in place and also returned).
    axes
        Cartesian axes to center along. Defaults to all three.

    Returns
    -------
    ase.Atoms
        Same object as ``atoms``, with its center of geometry at the cell center
        on the requested axes.
    """
    cell_center = np.array(atoms.cell).sum(axis=0) / 2.0
    com = atoms.get_positions().mean(axis=0)
    shift = np.zeros(3)
    for ax in axes:
        shift[ax] = cell_center[ax] - com[ax]
    atoms.translate(shift)
    return atoms


def add_vacuum(atoms: Atoms, vacuum: float, axes: Sequence[int]) -> Atoms:
    """Extend the cell along selected axes by adding vacuum padding.

    Parameters
    ----------
    atoms
        Structure to pad. The relevant cell vectors must be axis-aligned.
    vacuum
        Additional vacuum **per axis**, in Å (total padding, not half).
    axes
        Cartesian axes (0=x, 1=y, 2=z) to extend.

    Returns
    -------
    ase.Atoms
        Same object, with enlarged cell and re-centered coordinates along
        the padded axes. ``pbc`` on padded axes is preserved as set by caller.
    """
    cell = np.array(atoms.cell)
    for ax in axes:
        if abs(cell[ax, ax]) < 1e-6:
            cell[ax, ax] = 0.0
        cell[ax, ax] += vacuum
    atoms.set_cell(cell, scale_atoms=False)
    center_in_cell(atoms, axes=axes)
    return atoms


def minimum_image_distances(atoms: Atoms) -> np.ndarray:
    """Return the full pairwise distance matrix using the minimum-image
    convention.

    Uses :meth:`ase.Atoms.get_all_distances` with ``mic=True``. Safe for
    systems mixing periodic and non-periodic directions (ASE handles the mask
    from ``pbc``).

    Returns
    -------
    numpy.ndarray of shape (N, N)
        Symmetric distance matrix in Å.
    """
    return atoms.get_all_distances(mic=True)


def guess_bonds(
    atoms: Atoms,
    tolerance: float = 0.30,
    default_cutoff: float = MAX_CC_DISTANCE,
) -> list[tuple[int, int, float]]:
    """Guess covalent bonds from atomic positions.

    A pair ``(i, j)`` is considered bonded if their minimum-image distance is
    below ``r_i + r_j + tolerance`` using :data:`COVALENT_RADII`. If an
    element is unknown, ``default_cutoff`` is used.

    Parameters
    ----------
    atoms
        Structure to analyse.
    tolerance
        Extra slack (Å) on top of the sum of covalent radii.
    default_cutoff
        Fallback cutoff for unknown elements.

    Returns
    -------
    list of (i, j, distance) tuples, with ``i < j``.
    """
    from ase.neighborlist import neighbor_list

    if not len(atoms):
        return []

    # Cell list, not a full distance matrix. The pairwise matrix is O(N^2)
    # in both time and memory -- 24 s and 79 MB at 3136 atoms, and by
    # 28000 atoms (an ordinary MX2 coil) it wants 6 GB and never finishes.
    # Every export runs validation, so that quadratic was on the path of
    # every structure the framework produced.
    present = sorted(set(atoms.get_chemical_symbols()))
    cutoffs: dict[tuple[str, str], float] = {}
    for first in present:
        for second in present:
            override = BOND_CUTOFF_OVERRIDE.get((first, second))
            if override is not None:
                cutoffs[(first, second)] = override
                continue
            radius_a = COVALENT_RADII.get(first)
            radius_b = COVALENT_RADII.get(second)
            if radius_a is not None and radius_b is not None:
                cutoffs[(first, second)] = radius_a + radius_b + tolerance
            else:
                cutoffs[(first, second)] = default_cutoff

    first_index, second_index, distance = neighbor_list("ijd", atoms,
                                                        cutoff=cutoffs)
    # neighbor_list reports each pair twice, and a periodic self-image as
    # i == j; the 0.1 Å floor keeps a coincident pair from reading as a bond.
    keep = (first_index < second_index) & (distance > 0.1)
    return [
        (int(i), int(j), float(d))
        for i, j, d in zip(first_index[keep], second_index[keep],
                           distance[keep], strict=True)
    ]
