"""Graphene nanoribbon builder (armchair / zigzag edges).

Wraps :func:`ase.build.graphene_nanoribbon` and adds explicit vacuum in the
transverse and out-of-plane directions, so the structure is ready for DFT.
Optionally passivates edges with hydrogens.
"""

from __future__ import annotations

from typing import Any, Literal

from ase import Atoms
from ase.build import graphene_nanoribbon

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_2D
from ..utils.geometry import center_in_cell, guess_bonds
from .lattice_edits import apply_lattice_edits, periodic_box, ring_census

EdgeType = Literal["armchair", "zigzag"]


def build_nanoribbon(
    width: int,
    length: int,
    edge: EdgeType = "zigzag",
    bond: float = CC_BOND,
    vacuum: float = DEFAULT_VACUUM_2D,
    passivate: bool = False,
    defects: list[dict[str, Any]] | None = None,
    roughness: float = 0.0,
    seed: int | None = None,
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
    defects
        Stone-Wales rotations and reconstructed divacancies, as
        ``[{"type": "divacancy", "count": 1}]``; see
        :func:`~nanocarbon_lab.builders.lattice_edits.apply_lattice_edits`.
        Sites are kept a full ring in from either edge, so a defect never
        leaves an edge carbon on a stalk.
    roughness
        RMS out-of-plane corrugation in Å, 0 for the ideal flat sheet.
    seed
        Chooses the defect sites and the corrugation.

    Returns
    -------
    ase.Atoms
        Nanoribbon, periodic along z (ASE's own convention -- the axis its
        translational period runs along), non-periodic in x and y.

    Notes
    -----
    In ASE's convention the ribbon lies in the x-z plane and repeats
    along **z**. We preserve that and attach metadata so downstream code
    does not need to guess the axis.
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

    # ASE builds the ribbon in the x-z plane and marks z periodic, which
    # is the axis its repeat unit really runs along: for a zigzag ribbon
    # of 6 units the z edge is 14.76 Å, six times the 2.46 Å zigzag
    # period, while y is the out-of-plane vacuum. Overriding that to make
    # y the periodic axis -- which this did -- declared the ribbon
    # periodic across 15 Å of empty space and finite along its own length,
    # so every exported cell had its k-mesh on the wrong axis.
    atoms.set_pbc([False, False, True])
    center_in_cell(atoms, axes=(0, 1))

    bonds = [(i, j) for i, j, _ in guess_bonds(atoms)]
    atoms.info.update(
        {
            "structure_type": "nanoribbon",
            "edge": edge,
            "width": width,
            "length": length,
            "passivated": bool(passivate),
            "bond": bond,
            "periodic_axis": 2,
            # Measured, not assumed: a ribbon's hexagon count depends on
            # its edge and on whether the edges are passivated, and the
            # window has nothing to show without it.
            "bonds": [[int(i), int(j)] for i, j in bonds],
            # An open cylinder: periodic along its length, two free edges
            # across it. Euler characteristic 0, so no pentagons are owed
            # -- the curvature a cage pays for with twelve of them is
            # carried here by the boundary instead.
            "euler_expected": 0,
            "ring_counts": ring_census(atoms.get_positions(), bonds,
                                       periodic_box(atoms)),
        }
    )
    return apply_lattice_edits(atoms, defects=defects, roughness=roughness,
                               bond=bond, seed=seed)
