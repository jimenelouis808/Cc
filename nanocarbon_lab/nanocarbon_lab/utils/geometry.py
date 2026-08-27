"""Small geometry helpers built on top of :mod:`ase`.

The helpers here are deliberately backend-agnostic: they operate on
:class:`ase.Atoms` objects and return plain NumPy arrays or new :class:`ase.Atoms`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import Atoms

from .constants import COVALENT_RADII, MAX_CC_DISTANCE


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
    symbols = atoms.get_chemical_symbols()
    dmat = minimum_image_distances(atoms)
    n = len(atoms)
    bonds: list[tuple[int, int, float]] = []
    for i in range(n):
        ri = COVALENT_RADII.get(symbols[i])
        for j in range(i + 1, n):
            rj = COVALENT_RADII.get(symbols[j])
            if ri is not None and rj is not None:
                cutoff = ri + rj + tolerance
            else:
                cutoff = default_cutoff
            d = dmat[i, j]
            if 0.1 < d < cutoff:
                bonds.append((i, j, float(d)))
    return bonds
