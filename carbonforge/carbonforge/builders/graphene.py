"""Pristine graphene builders.

Provides the two-atom primitive cell and arbitrary orthogonal supercells,
always with a physically meaningful vacuum padding along z so the resulting
structure can be used directly as a 2D slab in DFT.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_2D


def build_graphene(bond: float = CC_BOND, vacuum: float = DEFAULT_VACUUM_2D) -> Atoms:
    """Build the primitive graphene unit cell (2 atoms, hexagonal lattice).

    The in-plane lattice constant is ``a = sqrt(3) * bond``. The cell is
    hexagonal in (x, y) and orthogonal along z, with vacuum for 2D use.

    Parameters
    ----------
    bond
        C-C bond length in Å.
    vacuum
        Total vacuum padding along z (Å).

    Returns
    -------
    ase.Atoms
        Two-atom primitive cell with ``pbc = (True, True, False)``.
    """
    a = np.sqrt(3.0) * bond
    cell = np.array(
        [
            [a, 0.0, 0.0],
            [a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, vacuum],
        ]
    )
    # Two-atom basis in fractional coords (1/3, 1/3) and (2/3, 2/3) of the
    # in-plane lattice, z at the middle of the vacuum. With a2 = a*(1/2, √3/2)
    # this gives a C-C distance of exactly ``bond``.
    frac = np.array([[1.0 / 3.0, 1.0 / 3.0, 0.5], [2.0 / 3.0, 2.0 / 3.0, 0.5]])
    positions = frac @ cell
    atoms = Atoms("C2", positions=positions, cell=cell, pbc=(True, True, False))
    atoms.info.update(
        {
            "structure_type": "graphene",
            "bond": bond,
            "lattice_a": a,
        }
    )
    return atoms


def build_graphene_supercell(
    nx: int = 1,
    ny: int = 1,
    bond: float = CC_BOND,
    vacuum: float = DEFAULT_VACUUM_2D,
    orthogonal: bool = True,
) -> Atoms:
    """Build an orthogonal (or hexagonal) graphene supercell.

    Parameters
    ----------
    nx, ny
        Number of repetitions along x and y. Must be positive integers.
    bond
        C-C bond length in Å.
    vacuum
        Total vacuum padding along z (Å).
    orthogonal
        If ``True`` (default), returns a rectangular supercell with 4 atoms
        per 1x1 orthogonal cell (a × a√3). Convenient for nanoribbons and
        LAMMPS boxes. If ``False``, repeats the primitive hexagonal cell.

    Returns
    -------
    ase.Atoms
        Graphene supercell.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1.")

    if not orthogonal:
        atoms = build_graphene(bond=bond, vacuum=vacuum) * (nx, ny, 1)
        atoms.info["structure_type"] = "graphene_supercell"
        return atoms

    a = np.sqrt(3.0) * bond
    cell = np.array(
        [
            [a, 0.0, 0.0],
            [0.0, a * np.sqrt(3.0), 0.0],
            [0.0, 0.0, vacuum],
        ]
    )
    # Four-atom orthogonal basis (conventional armchair-along-x choice).
    frac = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.5, 1.0 / 6.0, 0.5],
            [0.5, 0.5, 0.5],
            [0.0, 2.0 / 3.0, 0.5],
        ]
    )
    positions = frac @ cell
    unit = Atoms("C4", positions=positions, cell=cell, pbc=(True, True, False))
    atoms = unit * (nx, ny, 1)
    atoms.info.update(
        {
            "structure_type": "graphene_supercell",
            "bond": bond,
            "orthogonal": True,
            "nx": nx,
            "ny": ny,
        }
    )
    return atoms
