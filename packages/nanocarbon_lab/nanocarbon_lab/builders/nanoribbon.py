"""Graphene nanoribbon builder (armchair / zigzag edges).

Wraps :func:`ase.build.graphene_nanoribbon` and adds explicit vacuum in the
transverse and out-of-plane directions, so the structure is ready for DFT.
Optionally passivates edges with hydrogens.

The ribbon can also be **swept onto a centreline** -- a Z path, an S
curve, an arc -- and there the two ways to bend a ribbon are not
cosmetic variants of one another:

* **Out of plane** (``bend_plane="out_of_plane"``, the default) curls the
  sheet like a strip of paper. Every atom across the width rides the
  frame's *binormal*, which a rotation-minimizing frame does not rotate,
  so the deformation is an **isometry**: measured on a 360-atom ribbon
  folded 15.6 A over its 72 A length, every bond came out
  **1.4197-1.4200 A**, with no strain budget applied at all. That last
  2 parts in 10,000 is the sweep interpolating between a finite number
  of frames, not strain -- it shrinks with ``frame_samples``, where a
  strain would not. A ribbon can
  therefore take a sharp Z, which a swept tube cannot -- and the limit
  is not strain but the fold touching itself, so that is what is
  checked.
* **In plane** (``bend_plane="in_plane"``) bends the ribbon the hard
  way, stretching the outer edge and compressing the inner one. That is
  the tube's problem exactly, with the half-width in place of the tube
  radius, so it goes through the same
  :func:`centerline.fit_to_strain_budget`. Measured on the same ribbon
  and the same path, trimmed to the 8% budget: bonds 1.342-1.500 A for a
  2.4 A excursion. Six times less Z for bonds that are no longer ideal.

Sweeping also makes the ribbon **finite**. A straight ribbon is periodic
along its length and its bond list names pairs across that seam; a bent
one has no translational period, so the periodicity is dropped and the
bonds recomputed rather than carried over as lines the length of the
cell.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.build import graphene_nanoribbon

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_2D
from ..utils.geometry import center_in_cell, guess_bonds
from ..utils.rng import make_rng
from . import centerline as cl
from .assemblies import GRAPHITIC_GAP
from .lattice_edits import (
    apply_lattice_edits,
    periodic_box,
    ring_census,
    ring_faces,
)

EdgeType = Literal["armchair", "zigzag"]

#: How a swept ribbon bends. See the module docstring: ``out_of_plane``
#: is an isometry and ``in_plane`` is not.
BEND_PLANES = ("out_of_plane", "in_plane")

#: A fold that brings two parts of the sheet closer than this is
#: self-intersecting, not folded. Graphite's interlayer spacing is the
#: right floor: at that distance the two faces are in van der Waals
#: contact, which is as close as two graphitic sheets ever get.
MIN_FOLD_CLEARANCE = GRAPHITIC_GAP


def build_nanoribbon(
    width: int,
    length: int,
    edge: EdgeType = "zigzag",
    bond: float = CC_BOND,
    vacuum: float = DEFAULT_VACUUM_2D,
    passivate: bool = False,
    defects: list[dict[str, Any]] | None = None,
    roughness: float = 0.0,
    shape: cl.Shape = "straight",
    shape_turns: float = 1.0,
    waviness: float = 1.0,
    bend_plane: str = "out_of_plane",
    max_strain: float = cl.DEFAULT_MAX_STRAIN,
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
    shape
        Centreline to sweep the ribbon onto, one of
        :data:`centerline.Shape`. ``straight`` leaves the flat, periodic
        ribbon alone; anything else bends it and makes it **finite**.
        ``zigzag`` is the Z path.
    shape_turns
        Periods of a repeating shape (``zigzag``, ``helix``).
    waviness
        0-1 scale on the excursion. For ``out_of_plane`` this is honoured
        as asked, because the bend costs no strain; for ``in_plane`` it
        is the *request*, and the strain budget decides what is achieved.
    bend_plane
        One of :data:`BEND_PLANES`; see the module docstring for why the
        choice is physical rather than cosmetic.
    max_strain
        Outer-edge strain budget, used only by ``in_plane``.
    seed
        Chooses the defect sites and the corrugation.

    Returns
    -------
    ase.Atoms
        Nanoribbon, periodic along z (ASE's own convention -- the axis its
        translational period runs along), non-periodic in x and y. A
        **swept** ribbon is finite in all three, and carries
        ``shape``, ``bend_plane`` and the achieved excursion.

    Raises
    ------
    ValueError
        If ``bend_plane`` is unknown, or an out-of-plane fold brings the
        sheet within :data:`MIN_FOLD_CLEARANCE` of itself.

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
            # The census counted these and dropped the indices; the
            # window's ring colouring needs them, and without them a
            # ribbon came out one flat colour.
            "rings": ring_faces(atoms.get_positions(), bonds,
                                periodic_box(atoms)),
        }
    )
    atoms = apply_lattice_edits(atoms, defects=defects, roughness=roughness,
                                bond=bond, seed=seed)
    if shape == "straight":
        return atoms
    return _sweep_ribbon(atoms, shape=shape, turns=shape_turns,
                         waviness=waviness, bend_plane=bend_plane,
                         max_strain=max_strain, bond=bond, vacuum=vacuum,
                         seed=seed)


def _sweep_ribbon(
    atoms: Atoms,
    shape: cl.Shape,
    turns: float,
    waviness: float,
    bend_plane: str,
    max_strain: float,
    bond: float,
    vacuum: float,
    seed: int | None,
) -> Atoms:
    """Bend a flat, z-periodic ribbon onto a centreline. See the module
    docstring for why ``out_of_plane`` costs no strain and ``in_plane``
    does."""
    if bend_plane not in BEND_PLANES:
        raise ValueError(
            f"Unknown bend_plane {bend_plane!r}. Available: "
            f"{', '.join(BEND_PLANES)}.")
    if not 0.0 <= waviness <= 1.0:
        raise ValueError(f"waviness must be in [0, 1]; got {waviness}.")

    positions = atoms.get_positions()
    positions = positions - positions.mean(axis=0)
    half_width = 0.5 * float(np.ptp(positions[:, 0]))
    length = float(np.ptp(positions[:, 2]))
    control = cl.shape_control_points(
        shape, make_rng(seed), amplitude=waviness, turns=turns)

    if bend_plane == "in_plane":
        # The width rides the frame's NORMAL, which rotates with the
        # path, so the outer edge stretches: the tube's problem with the
        # half-width standing in for the tube radius. Same budget.
        control, strain = cl.fit_to_strain_budget(
            control, length, half_width, max_strain=max_strain)
        swept = cl.sweep_along_path(positions, control)
    else:
        # Hand the sweep the ribbon with its width on the BINORMAL, which
        # a rotation-minimizing frame does not rotate. Every atom across
        # the width then shares its cross-section's rigid motion, so the
        # sheet curls without a single bond changing length -- measured
        # 1.420-1.420 A. No budget is applied because there is nothing to
        # spend: an isometry has no strain to trim.
        strain = 0.0
        swept = cl.sweep_along_path(
            np.ascontiguousarray(positions[:, [1, 0, 2]]), control)
        swept = swept[:, [1, 0, 2]]

    # A fold has no strain to stop it, so what stops it is the sheet
    # meeting itself. Only pairs that are NOT near neighbours count: two
    # bonded atoms are 1.42 A apart by design.
    from scipy.spatial import cKDTree
    tree = cKDTree(swept)
    pairs = tree.query_pairs(MIN_FOLD_CLEARANCE, output_type="ndarray")
    if len(pairs):
        along = np.abs(positions[pairs[:, 0], 2] - positions[pairs[:, 1], 2])
        # Neighbours in the flat sheet are within a ring of each other;
        # a fold brings together atoms that were FAR apart along the
        # length, and that is the pair worth refusing.
        folded = pairs[along > 3.0 * bond]
        if len(folded):
            worst = float(np.linalg.norm(
                swept[folded[:, 0]] - swept[folded[:, 1]], axis=1).min())
            raise ValueError(
                f"This {shape} fold brings the sheet within {worst:.2f} A of "
                f"itself, under the {MIN_FOLD_CLEARANCE:.1f} A graphitic "
                "contact: the ribbon passes through itself rather than "
                "folding. An out-of-plane bend costs no strain, so nothing "
                "else limits it -- lower waviness, use fewer turns, or "
                "lengthen the ribbon.")

    bent = Atoms(symbols=atoms.get_chemical_symbols(), positions=swept,
                 pbc=(False, False, False))
    extents = np.ptp(swept, axis=0)
    bent.set_cell(np.diag(extents + vacuum))
    center_in_cell(bent, axes=(0, 1, 2))

    # The ribbon is no longer periodic, so its old bond list -- which
    # named pairs across the axial seam -- would draw lines the length of
    # the cell. Recompute from the bent geometry instead. This is the
    # sixth place in this package the same fact has bitten.
    bonds = [(i, j) for i, j, _ in guess_bonds(bent)]
    info = dict(atoms.info)
    info.update({
        "structure_type": "nanoribbon",
        "shape": shape,
        "bend_plane": bend_plane,
        "shape_turns": turns,
        "waviness": waviness,
        "swept": True,
        "periodic_axis": None,
        "path_strain": float(strain),
        # What the fold ACHIEVED, which for an in-plane bend is not what
        # was asked for. An in-plane bend swings the ribbon inside its
        # own plane, so its x span is the bend PLUS the width; report the
        # bend alone, so the number means the same thing in both modes.
        "excursion": float(
            np.ptp(swept[:, 1]) if bend_plane == "out_of_plane"
            else max(0.0, np.ptp(swept[:, 0]) - 2.0 * half_width)),
        "bonds": [[int(i), int(j)] for i, j in bonds],
        # Still an open sheet: two free edges, no translational period.
        # A disc rather than a cylinder now, so chi is 1, not 0.
        "euler_expected": 1,
        "ring_counts": ring_census(bent.get_positions(), bonds,
                                   periodic_box(bent)),
        "rings": ring_faces(bent.get_positions(), bonds,
                            periodic_box(bent)),
    })
    bent.info.update(info)
    return bent
