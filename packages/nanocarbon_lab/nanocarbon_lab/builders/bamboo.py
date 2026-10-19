"""Bamboo-like nanotubes: graphitic cups stacked inside a tube.

A bamboo CNT is a tube divided into compartments by internal walls, and
the one structural decision here is **what those walls are**. They are
not part of the tube's own carbon network, and that is geometry rather
than a modelling convenience:

**A sealed septum is not a manifold surface.** Where a septum meets the
wall, three sheets meet -- the wall above it, the wall below it, and the
septum itself -- so a point on that circle has three half-planes glued
along a line for a neighbourhood, which is not a disc. Marching cubes
returns the boundary of a solid, and a boundary is a manifold by
construction, so **the implicit route cannot produce a sealed septum at
any resolution or blend**. Said in carbon: an atom on that junction line
would need four neighbours in the network, and everything this package
meshes is a trivalent sp2 net.

Which is also what the samples show. The compartment walls of a
CVD-grown bamboo tube are **separate graphitic cups** held against the
wall by van der Waals, not a covalent continuation of it. So this is an
**assembly**, exactly as :mod:`assemblies` is for multi-wall tubes and
bundles -- and it costs nothing in wall quality, because both pieces
come from exact-lattice builders rather than from a remesher. The tube
is a rolled lattice and each cup is a sector cut, so every bond is
graphitic by construction and there is no amorphous wall anywhere.

The cup is :func:`nanocone.build_nanocone` at its one clean apex, a
single pentagon: ``n_pentagons=2`` and ``3`` put a square or a triangle
at the apex, which is sound topology and 1.339 / 1.230 A bonds, and
nature splits those disclinations into separate pentagons instead. The
cone is **developable**, so unrolling it is an isometry and its bonds
are 1.42 by construction -- the same reason the package prefers a
crystalline route wherever one exists.

What this does NOT claim: nothing here says the cups sit where a real
growth would put them. Their spacing is an input, the compartment
length in a real sample being set by the catalyst's feeding cycle,
which nothing in this framework models.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from ..utils.geometry import center_in_cell
from .assemblies import GRAPHITIC_GAP
from .cnt import build_cnt
from .nanocone import build_nanocone

__all__ = ["CUP_ORIENTATIONS", "build_bamboo_cnt"]

#: Which way the cups point. ``up`` is every cup the same way, which is
#: what a single growth direction gives; ``alternate`` flips every other
#: one, which some samples show where growth reversed.
CUP_ORIENTATIONS = ("up", "down", "alternate")

#: A cup must clear the wall by a graphitic contact or it is inside it.
#: The same gap the multi-wall builder holds its shells at, for the
#: same reason: this is a van der Waals contact and not a bond.
MIN_CUP_CLEARANCE = GRAPHITIC_GAP


def build_bamboo_cnt(
    n: int = 16,
    m: int = 16,
    n_compartments: int = 3,
    compartment_length: float = 20.0,
    cup_pentagons: int = 1,
    orientation: str = "up",
    cup_clearance: float = GRAPHITIC_GAP,
    bond: float = CC_BOND,
    vacuum: float = DEFAULT_VACUUM_1D,
    seed: int | None = 0,
) -> Atoms:
    """Build a bamboo-like nanotube: a tube with graphitic cups inside.

    Parameters
    ----------
    n, m
        Chiral indices of the outer tube. ``(n, 0)`` is zigzag,
        ``(n, n)`` armchair. **A bamboo tube has to be wide**, and that
        is the same kind of fact as the nanocoil's radius rather than a
        default to taste: the cup has to fit inside the wall at a
        graphitic contact, so a (10,10) tube -- radius 6.78 A -- leaves
        3.38 A against a 3.40 A clearance and cannot hold one at all.
        Measured rim room: (12,12) 4.74 A, (14,14) 6.09, **(16,16)
        7.45** (the default), (20,20) 10.16. Real bamboo tubes are
        multi-wall and tens of nm across, which is the same answer from
        the sample's side.
    n_compartments
        Compartments in the periodic cell, which carries that many cups:
        the divider at the far face is the one at the near face.
    compartment_length
        Target axial length of one compartment (Å). The lattice
        quantises the tube's period, so the achieved value is an
        **output** and is reported as ``info["compartment_length"]``
        beside the request. Neither is a prediction: what sets it in a
        real sample is the catalyst's feeding cycle, which nothing here
        models.
    cup_pentagons
        Apex disclinations of each cup. **1 is the only clean one** --
        470 atoms at 1.391-1.420 Å -- while 2 and 3 put a square or a
        triangle at the apex and are refused here rather than returned
        with 1.230 Å bonds.
    orientation
        One of :data:`CUP_ORIENTATIONS`.
    cup_clearance
        Van der Waals gap (Å) between a cup's rim and the tube wall.

    Returns
    -------
    ase.Atoms
        Tube plus cups as one structure, with ``info["pieces"]`` giving
        each piece's atom range so a consumer can tell them apart. The
        bond list carries **no** bond between a cup and the wall,
        because there is none: they are in contact, not bonded.

    Raises
    ------
    ValueError
        If the apex is not the clean one, or the tube is too narrow to
        hold a cup at a graphitic clearance.
    """
    if n_compartments < 1:
        raise ValueError(
            "A bamboo tube needs at least 1 compartment, which is one "
            "cup. With none it is a plain nanotube -- use the `cnt` mode.")
    if cup_pentagons != 1:
        raise ValueError(
            f"cup_pentagons={cup_pentagons} puts a "
            f"{'square' if cup_pentagons == 2 else 'triangle or worse'} at "
            "the apex: sound topology and 1.34/1.23 A bonds, which is not "
            "carbon. Nature splits those disclinations into separate "
            "pentagons instead. Only 1 is clean here.")
    if compartment_length <= 0.0:
        raise ValueError("compartment_length must be positive.")

    # The tube is PERIODIC along its axis, and its bond list records
    # pairs across that seam. Assembling it into a finite structure
    # draws those as lines the length of the cell -- measured, a 61.5 A
    # "bond" against a real 1.42. This is the fifth place in this
    # package the same fact has bitten, so the structure stays periodic
    # along z and the seam bonds stay true.
    #
    # A periodic cell of n compartments therefore carries n cups, not
    # n - 1: the divider at the far face IS the one at the near face.
    n_cups = n_compartments
    tube = build_cnt(n=n, m=m, length=n_compartments * compartment_length,
                     bond=bond, vacuum=vacuum)
    # The lattice quantises the axial period, so the compartment length
    # is an OUTPUT. Report what was achieved, never what was asked.
    period = float(tube.cell[2, 2])
    achieved_compartment = period / n_compartments
    tube_positions = tube.get_positions()
    axis_centre = tube_positions[:, :2].mean(axis=0)
    radial = np.linalg.norm(tube_positions[:, :2] - axis_centre, axis=1)
    tube_radius = float(radial.mean())

    # The cup has to fit INSIDE the wall, which fixes its rim radius.
    # A cone's rim radius is its slant radius times sin(theta/2), and
    # for the clean one-pentagon apex sin(theta/2) = 1 - 1/6 = 5/6.
    rim_target = tube_radius - cup_clearance
    if rim_target <= MIN_CUP_CLEARANCE:
        raise ValueError(
            f"A ({n},{m}) tube has radius {tube_radius:.2f} A, which "
            f"leaves {rim_target:.2f} A inside a {cup_clearance:.2f} A "
            "clearance -- not enough for a cup. Use a wider tube.")
    slant = rim_target / (1.0 - cup_pentagons / 6.0)

    positions: list[np.ndarray] = [tube_positions]
    bonds: list[list[int]] = list(map(list, tube.info.get("bonds", [])))
    rings: list[list[int]] = [list(ring)
                              for ring in tube.info.get("rings", [])]
    ring_counts: dict[int, int] = dict(tube.info.get("ring_counts", {}))
    pieces: list[tuple[str, int, int]] = [("tube", 0, len(tube))]
    offset = len(tube)

    axial_low = float(tube_positions[:, 2].min())
    cup = build_nanocone(n_pentagons=cup_pentagons, radius=slant, bond=bond)
    cup_positions = cup.get_positions()
    cup_positions = cup_positions - cup_positions.mean(axis=0)

    for index in range(n_cups):
        if orientation == "up":
            flip = False
        elif orientation == "down":
            flip = True
        elif orientation == "alternate":
            flip = bool(index % 2)
        else:
            raise ValueError(
                f"Unknown orientation {orientation!r}. Available: "
                f"{', '.join(CUP_ORIENTATIONS)}.")
        placed = cup_positions.copy()
        if flip:
            placed[:, 2] = -placed[:, 2]
        placed[:, 0] += axis_centre[0]
        placed[:, 1] += axis_centre[1]
        placed[:, 2] += axial_low + index * achieved_compartment

        positions.append(placed)
        # The cup's own bonds and rings move with it. Nothing joins a
        # cup to the wall, because nothing does in the sample either.
        bonds += [[a + offset, b + offset] for a, b in cup.info["bonds"]]
        rings += [[a + offset for a in ring] for ring in cup.info["rings"]]
        for size, count in cup.info["ring_counts"].items():
            ring_counts[int(size)] = ring_counts.get(int(size), 0) + count
        pieces.append(("cup", offset, offset + len(cup)))
        offset += len(cup)

    merged = np.vstack(positions)
    atoms = Atoms(symbols=["C"] * len(merged), positions=merged,
                  pbc=(False, False, True))
    # Transverse axes get vacuum; the periodic one keeps the lattice's
    # own period, because padding it would change the crystal.
    extents = merged.max(axis=0) - merged.min(axis=0)
    atoms.set_cell(np.diag([extents[0] + vacuum, extents[1] + vacuum,
                            period]))
    center_in_cell(atoms, axes=(0, 1))

    atoms.info.update({
        "structure_type": "bamboo_cnt",
        "n": n,
        "m": m,
        "n_compartments": n_compartments,
        "n_cups": n_cups,
        "compartment_length": achieved_compartment,
        "compartment_length_requested": compartment_length,
        "axial_period": period,
        "cup_pentagons": cup_pentagons,
        "cup_orientation": orientation,
        "cup_clearance": cup_clearance,
        "tube_radius": tube_radius,
        "cup_rim_radius": rim_target,
        "cup_slant_radius": slant,
        "bond": bond,
        "pieces": pieces,
        "ring_counts": {int(k): int(v) for k, v in ring_counts.items()},
        "rings": rings,
        "bonds": bonds,
        "note": (
            "The cups are separate graphitic pieces held against the wall "
            "by van der Waals, not a covalent part of it: a sealed septum "
            "is not a manifold surface, so no closed carbon network has "
            "one. Compartment length is an input, not a prediction."),
    })
    return atoms
