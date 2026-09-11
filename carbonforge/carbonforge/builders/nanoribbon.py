"""Graphene nanoribbon builder (armchair / zigzag edges).

Wraps :func:`ase.build.graphene_nanoribbon` and adds explicit vacuum in the
transverse and out-of-plane directions, so the structure is ready for DFT.
Optionally passivates edges with hydrogens.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms
from ase.build import graphene_nanoribbon

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_2D
from ..utils.geometry import center_in_cell


EdgeType = Literal["armchair", "zigzag"]


def build_nanoribbon(
    width: int,
    length: int,
    edge: EdgeType = "zigzag",
    bond: float = CC_BOND,
    vacuum: float = DEFAULT_VACUUM_2D,
    passivate: bool = False,
) -> Atoms:
    """Build a graphene nanoribbon.

    Parameters
    ----------
    width
        Ribbon width in number of dimer lines (edge-dependent convention
        used by ASE).
    length
        Number of repeat units along the periodic axis.
    edge
        ``"zigzag"`` or ``"armchair"``.
    bond
        C-C bond length (Å).
    vacuum
        Total vacuum along the two non-periodic axes (Å).
    passivate
        If ``True``, saturate edge carbons with hydrogen (C-H = 1.09 Å).

    Returns
    -------
    ase.Atoms
        Nanoribbon, periodic along y (ASE convention), non-periodic in x/z.

    Notes
    -----
    ASE lays the ribbon in the **x-z plane**: width along x, ribbon axis along
    **z**, and vacuum along y (out of plane). So the periodic direction is z,
    and ASE sets ``pbc = (False, False, True)`` itself — which this builder
    keeps rather than overriding.

    Getting this wrong is not cosmetic. Declaring the wrong periodic axis
    makes the band path run through vacuum, has the k-mesh sample the empty
    direction while treating the real one as isolated, and points the vacuum
    check at the wrong axes. Every ribbon export would be physically wrong
    while looking perfectly well-formed.
    """
    if width < 1 or length < 1:
        raise ValueError("width and length must be >= 1.")

    atoms = graphene_nanoribbon(
        width,
        length,
        type=edge,
        saturated=passivate,
        C_H=1.09,
        C_C=bond,
        vacuum=vacuum / 2.0,
        sheet=False,
    )

    # ASE already sets pbc = (False, False, True) and pads x and y. Trust it;
    # only re-centre the atoms within the two non-periodic directions.
    center_in_cell(atoms, axes=(0, 1))

    atoms.info.update(
        {
            "structure_type": "nanoribbon",
            "edge": edge,
            "width": width,
            "length": length,
            "passivated": bool(passivate),
            "bond": bond,
            "periodic_axis": 2,
        }
    )
    return atoms
