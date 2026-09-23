"""Graphene nanoribbon builder (armchair / zigzag edges).

Wraps :func:`ase.build.graphene_nanoribbon` and adds explicit vacuum in the
transverse and out-of-plane directions, so the structure is ready for DFT.
Optionally passivates edges with hydrogens.

:func:`build_finite_nanoribbon` cuts the same ribbon to a finite, fully
hydrogen-terminated flake with vacuum on every side. That is the model to use
whenever the calculation needs a dipole moment -- IR intensities by finite
differences, above all -- because the dipole of a system that is periodic
along one axis is not defined along that axis.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms
from ase.build import graphene_nanoribbon

from ..functionalization.attach import passivate_edges
from ..topology.graph import build_bond_graph
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


#: Vacuum on each side of a finite flake, in Å. 6 Å per side (12 Å between
#: periodic images) is the least a neutral, weakly polar molecule tolerates
#: before the images start to talk through the density tail and the dipole.
MIN_VACUUM_PER_SIDE: float = 6.0
DEFAULT_VACUUM_PER_SIDE: float = 7.0


def rebox(atoms: Atoms, vacuum_per_side: float = DEFAULT_VACUUM_PER_SIDE) -> Atoms:
    """Put a finite structure in an orthorhombic box with equal vacuum on every side.

    Modifies ``atoms`` in place and returns it. Every axis is made
    non-periodic.
    """
    positions = atoms.get_positions()
    low = positions.min(axis=0)
    span = positions.max(axis=0) - low
    atoms.set_cell(np.diag(span + 2.0 * vacuum_per_side), scale_atoms=False)
    atoms.pbc = False
    # Centre the extent, not the centroid: a group on one edge moves the
    # centroid and would leave less vacuum on that side.
    atoms.translate(vacuum_per_side - low)
    return atoms


def build_finite_nanoribbon(
    width: int,
    length: int,
    edge: EdgeType = "armchair",
    bond: float = CC_BOND,
    vacuum_per_side: float = DEFAULT_VACUUM_PER_SIDE,
    passivate: bool = True,
) -> Atoms:
    """Build a finite graphene nanoribbon (a flake) with every edge terminated.

    Parameters
    ----------
    width, length, edge, bond
        As in :func:`build_nanoribbon`. ``length`` counts repeat units along
        the ribbon axis, which here becomes the flake's long dimension.
    vacuum_per_side
        Vacuum between the outermost atom and the cell boundary on **each**
        side of **each** axis, in Å. At least :data:`MIN_VACUUM_PER_SIDE`.
    passivate
        Terminate every two-coordinated carbon with hydrogen -- the long
        edges and the two ends. A bare edge carbon is a radical, so leaving
        them is a different chemical system, not a cheaper version of the
        same one.

    Returns
    -------
    ase.Atoms
        Non-periodic along all three axes. The flake lies in the x-z plane
        with its long axis along z (the ASE ribbon convention), so y is the
        plane normal.

    Notes
    -----
    The ends have the *other* edge type: an armchair ribbon ends in zigzag
    edges and a zigzag ribbon in armchair ones. Zigzag edges carry localised,
    spin-polarised states, so both flavours of flake can need a
    spin-polarised calculation; see :mod:`carbonforge.vibspec.core.checks`.
    """
    if vacuum_per_side < MIN_VACUUM_PER_SIDE:
        raise ValueError(
            f"vacuum_per_side={vacuum_per_side} Å es poco: con menos de "
            f"{MIN_VACUUM_PER_SIDE} Å por lado la molécula interactúa con sus "
            "imágenes periódicas."
        )

    atoms = build_nanoribbon(width, length, edge=edge, bond=bond, passivate=False)
    atoms.pbc = False

    # Cutting the periodic bonds can leave a carbon with a single neighbour
    # at the ends. It cannot be passivated into anything sensible (it would
    # be a CH2 sticking out of an aromatic edge), so it goes.
    while True:
        graph = build_bond_graph(atoms)
        loose = [i for i in range(len(atoms)) if graph.degree[i] < 2]
        if not loose:
            break
        del atoms[loose]

    n_hydrogen = 0
    if passivate:
        before = len(atoms)
        atoms = passivate_edges(atoms, "H")
        n_hydrogen = len(atoms) - before
        # Hydrogen termination is part of the flake, not a functional group:
        # keep it out of the record that coverage() counts.
        atoms.info.pop("functionalization", None)

    rebox(atoms, vacuum_per_side)
    atoms.info.update(
        {
            "structure_type": "finite_nanoribbon",
            "edge": edge,
            "end_edge": "zigzag" if edge == "armchair" else "armchair",
            "width": width,
            "length": length,
            "passivated": bool(passivate),
            "n_hydrogen": n_hydrogen,
            "bond": bond,
            "ribbon_axis": 2,
            "plane_normal": 1,
            "vacuum_per_side": vacuum_per_side,
        }
    )
    atoms.info.pop("periodic_axis", None)
    return atoms
