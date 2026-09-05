"""Turn any structure in this package into a periodic unit cell.

Every plane-wave DFT code -- Quantum ESPRESSO, VASP, CASTEP -- and every
periodic viewer -- VESTA, OVITO, XCrySDen -- is **three-dimensionally
periodic**. There is no "molecule" setting: a molecule is a molecule in a
box big enough that it does not see its own images. So the question
"what is the unit cell of this structure?" always has an answer, and the
answer depends on how many directions the structure genuinely repeats
in:

======  ===================================  =========================
starts  examples                             becomes
======  ===================================  =========================
0D      fullerene, capped tube, junction     molecule in a box
1D      (n, m) nanotube                      tube period + vacuum
2D      graphene, MX2 layer, twisted cell    slab + vacuum along z
3D      schwarzite, nanotube network, bulk   already a unit cell
======  ===================================  =========================

The conversion itself is a few lines. What makes it worth a module is
everything around it:

* **The cell must be honest about what is periodic.** A structure with
  ``pbc=(True, True, False)`` and a z cell length is ambiguous -- the
  exporters have to guess whether z is vacuum or a repeat. After this
  every axis is periodic and the vacuum is real, which is what the codes
  above actually consume.
* **Vacuum is measured, not assumed** -- and measured only where it is
  vacuum. The number that matters is the nearest approach to a
  neighbouring image *across a direction the structure does not repeat
  in*. Counting every image instead is precisely backwards: in a real
  crystal an atom bonds to its image, so a nanotube's 1.42 Å contact
  along its own axis is the structure, not a convergence failure.
* **Atoms must sit inside the cell.** A periodic viewer draws the box
  and the contents; atoms outside it look like the structure has burst
  its cell, and some codes reject them outright.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import Atoms

from .utils.constants import DEFAULT_VACUUM_1D, DEFAULT_VACUUM_2D

#: Image separation below which a molecule-in-a-box is too tight to trust.
#: Two neighbouring images of a neutral carbon structure stop interacting
#: at roughly twice the van der Waals contact; 8 Å is the usual working
#: minimum and 10-12 Å the comfortable one.
MIN_IMAGE_SEPARATION = 8.0


def periodicity(atoms: Atoms) -> int:
    """How many directions the structure genuinely repeats in (0-3)."""
    return int(np.count_nonzero(atoms.get_pbc()))


def describe_periodicity(atoms: Atoms) -> str:
    """``"0D"`` … ``"3D"``, from the structure's own ``pbc`` flags."""
    return f"{periodicity(atoms)}D"


def image_separation(atoms: Atoms, vacuum_axes: Sequence[int] | None = None) -> float:
    """Shortest distance across a **vacuum** direction to a neighbouring image.

    This is the number that says whether a cell is big enough, and
    getting it right means being clear about which images are supposed
    to be close. In a genuine crystal an atom *bonds* to its image: a
    nanotube's periodic axis puts carbons 1.42 Å apart across the
    boundary, and a schwarzite's puts them at 1.37 Å. Measuring every
    image and reporting the minimum called both of those unconverged,
    which is precisely backwards -- that contact is the structure.

    So only images displaced along a direction the structure does *not*
    repeat in are counted. Along those, nothing should be in contact,
    and how close the nearest approach gets is exactly the convergence
    question a plane-wave calculation is asking.

    Parameters
    ----------
    atoms
        Structure to measure.
    vacuum_axes
        Which axes are vacuum rather than genuine repeats. Defaults to
        the axes recorded by :func:`to_unit_cell`, falling back to the
        structure's own non-periodic axes.

    Returns
    -------
    float
        The nearest approach across vacuum, or ``inf`` when there is no
        vacuum direction at all -- a bulk crystal has nothing to
        converge, and reporting a number there would invite it to be
        compared against a threshold that does not apply.
    """
    if vacuum_axes is None:
        recorded = atoms.info.get("unit_cell", {}).get("vacuum_axes")
        if recorded:
            vacuum_axes = [int(axis) for axis in recorded]
        else:
            vacuum_axes = [axis for axis in range(3)
                           if not atoms.get_pbc()[axis]]
    vacuum_axes = list(vacuum_axes)
    if not vacuum_axes or not any(atoms.get_pbc()):
        return float("inf")

    from ase.neighborlist import neighbor_list

    # Generous: anything past this is comfortably converged, and asking
    # for more only makes the neighbour list bigger.
    cutoff = 20.0
    _, _, distance, offset = neighbor_list("ijdS", atoms, cutoff=cutoff)
    crosses = np.any(offset[:, vacuum_axes] != 0, axis=1)
    if not np.any(crosses):
        return cutoff
    return float(distance[crosses].min())


def to_unit_cell(
    atoms: Atoms,
    vacuum: float | None = None,
    wrap: bool = True,
) -> Atoms:
    """Return a copy that is a fully periodic, DFT-ready unit cell.

    Periodic directions are left exactly as they are -- their lattice
    vector *is* the physics and must not be padded. Non-periodic
    directions get ``vacuum`` of empty space on each side and are then
    marked periodic, which is what a plane-wave code needs in order to
    treat them as vacuum rather than guess.

    Parameters
    ----------
    atoms
        Any structure from this package (not mutated).
    vacuum
        Padding in Å added on **each** side of a non-periodic direction,
        so the axis grows by ``2 * vacuum``. Defaults to
        :data:`~nanocarbon_lab.utils.constants.DEFAULT_VACUUM_2D` (15 Å)
        for a structure with a single non-periodic axis -- a slab, where
        the images stack face to face -- and
        :data:`~nanocarbon_lab.utils.constants.DEFAULT_VACUUM_1D` (12 Å)
        otherwise.
    wrap
        Fold atoms into the cell. On by default: atoms drawn outside the
        box are the commonest reason a correct periodic structure looks
        broken in a viewer.

    Returns
    -------
    ase.Atoms
        ``pbc=(True, True, True)``, with a ``unit_cell`` entry in
        ``info`` recording what it started from, the padding applied per
        axis and the achieved image separation.

    Raises
    ------
    ValueError
        If the structure has no atoms, or a periodic axis has no lattice
        vector to keep.
    """
    if len(atoms) == 0:
        raise ValueError("Cannot build a unit cell for an empty structure.")

    original = describe_periodicity(atoms)
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    if vacuum is None:
        # A slab's images stack face to face across the one open
        # direction, so it wants more room than a tube, whose two open
        # directions each see a thinner object.
        vacuum = DEFAULT_VACUUM_2D if int(pbc.sum()) == 2 else DEFAULT_VACUUM_1D
    if vacuum < 0:
        raise ValueError("vacuum must be >= 0.")

    out = atoms.copy()
    out.info = {**atoms.info}
    cell = np.array(out.cell, dtype=float)
    positions = out.get_positions()

    padded: dict[int, float] = {}
    for axis in range(3):
        if pbc[axis]:
            if not np.any(cell[axis]):
                raise ValueError(
                    f"Axis {axis} is marked periodic but has no lattice "
                    "vector; the structure's cell is inconsistent."
                )
            continue
        # Rebuild this axis from the atoms rather than trusting whatever
        # box the builder happened to leave: a finite builder's cell is a
        # bounding box with padding already in it, and padding a padded
        # box compounds the error every time this is called.
        span = float(positions[:, axis].max() - positions[:, axis].min())
        length = span + 2.0 * vacuum
        cell[axis] = 0.0
        cell[axis, axis] = length
        padded[axis] = vacuum

    out.set_cell(cell)
    out.set_pbc(True)

    # Centre only along the axes that were rebuilt. Shifting a periodic
    # axis is harmless but pointless, and doing it would move atoms
    # relative to a lattice that is already correct.
    if padded:
        centre = np.array(out.cell.array).sum(axis=0) / 2.0
        shift = centre - positions.mean(axis=0)
        for axis in range(3):
            if axis not in padded:
                shift[axis] = 0.0
        out.set_positions(positions + shift)

    if wrap:
        out.wrap()

    separation = image_separation(out, vacuum_axes=list(padded))
    out.info["unit_cell"] = {
        "original_periodicity": original,
        "vacuum_axes": {axis: round(value, 3) for axis, value in padded.items()},
        "periodic_axes": [axis for axis in range(3) if pbc[axis]],
        "lengths": [round(float(value), 4) for value in out.cell.lengths()],
        "angles": [round(float(value), 3) for value in out.cell.angles()],
        "volume": round(float(out.cell.volume), 3),
        "image_separation": round(separation, 3)
        if np.isfinite(separation) else None,
        "converged": bool(separation >= MIN_IMAGE_SEPARATION),
    }
    return out


def cell_report(atoms: Atoms) -> dict:
    """Measured description of a structure's cell, for printing.

    Reports the achieved image separation alongside the lattice, because
    the lattice alone does not say whether the cell is big enough and
    that is the only question a person is really asking.
    """
    separation = image_separation(atoms)
    return {
        "periodicity": describe_periodicity(atoms),
        "pbc": tuple(bool(value) for value in atoms.get_pbc()),
        "lengths": tuple(round(float(v), 4) for v in atoms.cell.lengths()),
        "angles": tuple(round(float(v), 3) for v in atoms.cell.angles()),
        "volume": round(float(atoms.cell.volume), 3),
        "n_atoms": len(atoms),
        "density": (round(sum(atoms.get_masses()) / atoms.cell.volume * 1.66054, 4)
                    if atoms.cell.volume > 0 else None),
        "image_separation": round(separation, 3) if np.isfinite(separation) else None,
        "converged": bool(separation >= MIN_IMAGE_SEPARATION),
        "atoms_outside": int(_n_outside(atoms)),
    }


def _n_outside(atoms: Atoms) -> int:
    """How many atoms sit outside the cell, in fractional coordinates.

    Not cosmetic: a periodic viewer draws the box and its contents, so
    atoms outside make a correct structure look like it has burst.
    """
    if atoms.cell.rank < 3:
        return 0
    fractional = atoms.cell.scaled_positions(atoms.get_positions())
    return int(np.count_nonzero((fractional < -1e-6) | (fractional > 1 + 1e-6)))


__all__ = [
    "MIN_IMAGE_SEPARATION",
    "cell_report",
    "describe_periodicity",
    "image_separation",
    "periodicity",
    "to_unit_cell",
]
