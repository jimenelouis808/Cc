"""Carbon nanotube builders (armchair, zigzag, chiral).

The CNT math (chiral vector, translational period) is delegated to
:func:`ase.build.nanotube`, which is well tested and produces correct
periodic tubes for any (n, m). On top of that we:

* enforce sensible vacuum in the two transverse directions,
* set ``pbc`` consistently with the 1D periodicity,
* center the tube in the cell,
* expose a single :func:`build_cnt` entry point covering armchair/zigzag/chiral.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms
from ase.build import nanotube

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from ..utils.geometry import center_in_cell


Chirality = Literal["armchair", "zigzag", "chiral"]


def _infer_chirality(n: int, m: int) -> Chirality:
    """Return the chirality family of an ``(n, m)`` CNT."""
    if m == 0:
        return "zigzag"
    if n == m:
        return "armchair"
    return "chiral"


def build_cnt(
    n: int,
    m: int,
    length: float = 10.0,
    bond: float = CC_BOND,
    vacuum: float = DEFAULT_VACUUM_1D,
    axis: int = 2,
) -> Atoms:
    """Build a single-wall carbon nanotube ``(n, m)``.

    The tube is periodic along ``axis`` and padded with vacuum on the two
    transverse directions so it can be used directly in DFT or MD.

    Parameters
    ----------
    n, m
        Chiral indices. Must satisfy ``n >= 1`` and ``0 <= m <= n``.
        ``(n, 0)`` is zigzag, ``(n, n)`` is armchair, everything else is chiral.
    length
        Target tube length in Å. The builder returns an integer number of
        translational periods whose total length is at least ``length``.
    bond
        C-C bond length in Å. Default 1.42 (sp2 equilibrium).
    vacuum
        Total transverse vacuum padding in Å (applied symmetrically).
    axis
        Cartesian axis along which the tube is periodic (0, 1 or 2). The
        two remaining axes receive the vacuum padding.

    Returns
    -------
    ase.Atoms
        Carbon nanotube with ``pbc[axis] = True`` and
        ``pbc[other] = False``.

    Raises
    ------
    ValueError
        For non-physical ``(n, m)`` or ``length <= 0``.
    """
    if n < 1 or m < 0 or m > n:
        raise ValueError(f"Invalid chirality (n={n}, m={m}): require n>=1 and 0<=m<=n.")
    if length <= 0:
        raise ValueError("length must be positive.")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2.")

    # One translational period.
    unit = nanotube(n, m, length=1, bond=bond, symbol="C", verbose=False)
    period = float(unit.cell[2, 2])
    n_periods = max(1, int(np.ceil(length / period)))

    atoms = nanotube(n, m, length=n_periods, bond=bond, symbol="C", verbose=False)

    # ASE builds the tube along z; permute axes if the user asks for x or y.
    if axis != 2:
        perm = [0, 1, 2]
        perm[axis], perm[2] = perm[2], perm[axis]
        positions = atoms.get_positions()[:, perm]
        cell = np.array(atoms.cell)[:, perm][perm, :]
        atoms.set_positions(positions)
        atoms.set_cell(cell, scale_atoms=False)

    # Diameter from the periodic-axis-orthogonal positions.
    other = [i for i in (0, 1, 2) if i != axis]
    pos = atoms.get_positions()
    radius = float(np.linalg.norm(pos[:, other] - pos[:, other].mean(axis=0), axis=1).max())

    # Transverse cell = 2*radius + vacuum.
    cell = np.array(atoms.cell)
    box = 2.0 * radius + vacuum
    for ax in other:
        cell[ax, :] = 0.0
        cell[ax, ax] = box
    atoms.set_cell(cell, scale_atoms=False)

    pbc = [False, False, False]
    pbc[axis] = True
    atoms.set_pbc(pbc)

    center_in_cell(atoms, axes=other)

    atoms.info.update(
        {
            "structure_type": "CNT",
            "chirality": _infer_chirality(n, m),
            "n": n,
            "m": m,
            "n_periods": n_periods,
            "period_length": period,
            "radius": radius,
            "bond": bond,
            "axis": axis,
        }
    )
    return atoms
