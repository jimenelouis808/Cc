"""Small geometry helpers built on top of :mod:`ase`.

The helpers here are deliberately backend-agnostic: they operate on
:class:`ase.Atoms` objects and return plain NumPy arrays or new :class:`ase.Atoms`.
"""

from __future__ import annotations

import itertools
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


def bond_shifts(atoms: Atoms, bonds: Sequence[Sequence[int]]) -> np.ndarray:
    """Which lattice translation each bond crosses, in whole cell units.

    A bond list built under the minimum-image convention records the pair
    ``(i, j)`` and nothing else, so a bond between an atom on one face of
    the cell and its neighbour on the opposite face looks, in raw
    coordinates, like a 30 Å bond straight across the structure. It is not:
    the neighbour is one cell away, and the bond is 1.42 Å like every
    other. This returns the integer translation **s** for each bond such
    that

        ``positions[j] - s @ cell - positions[i]``

    is the short vector — zero for an ordinary interior bond, and a unit
    vector along one axis for a bond that leaves through a face.

    Two things need it. Drawing: without the shift a periodic structure is
    covered in long straight lines from edge to edge, which is the single
    biggest source of visual noise in a schwarzite or a coil. And
    repetition: a supercell's bond list is the original one re-indexed
    copy by copy, and knowing which cell each bond lands in is exactly
    what says which copy the far end belongs to.

    The search is a candidate scan rather than a rounding. Projecting the
    bond vector onto the periodic sub-lattice and rounding is right for an
    orthogonal cell and can be off by one for a strongly sheared one, so
    the rounded value is used as a centre and the neighbouring integers
    are tried too. The winner is the translation that makes the bond
    shortest, which is the definition.

    Parameters
    ----------
    atoms
        The structure. Only its cell and ``pbc`` are read.
    bonds
        Pairs of atom indices.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(len(bonds), 3)``. All zeros when the
        structure is not periodic, so callers need no special case.
    """
    pairs = np.asarray(bonds, dtype=int).reshape(-1, 2)
    shifts = np.zeros((len(pairs), 3), dtype=int)
    if not len(pairs):
        return shifts

    cell = np.asarray(atoms.cell, dtype=float)
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    axes = [i for i in range(3)
            if pbc[i] and float(np.linalg.norm(cell[i])) > 1e-6]
    if not axes:
        return shifts

    lattice = cell[axes]                                   # (k, 3)
    gram = lattice @ lattice.T
    try:
        inverse = np.linalg.inv(gram)
    except np.linalg.LinAlgError:                          # degenerate cell
        return shifts

    positions = atoms.get_positions()
    delta = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    centre = np.rint(delta @ lattice.T @ inverse).astype(int)   # (n, k)

    # One step either side of the projection, along each periodic axis.
    steps = np.array(list(itertools.product((-1, 0, 1), repeat=len(axes))),
                     dtype=int)
    best = None
    best_length = None
    for step in steps:
        trial = centre + step
        length = np.linalg.norm(delta - trial @ lattice, axis=1)
        if best is None:
            best, best_length = trial, length
            continue
        better = length < best_length
        best = np.where(better[:, None], trial, best)
        best_length = np.where(better, length, best_length)

    shifts[:, axes] = best
    return shifts
