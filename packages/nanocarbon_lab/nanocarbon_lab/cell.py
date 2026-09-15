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
    mark: str = "3D",
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
    mark
        Which axes come back flagged periodic, once the vacuum is in
        place. The geometry is identical either way; only ``pbc`` differs.

        ``"3D"`` (default) flags all three, which is what a plane-wave
        code is given and what formats like CIF can express at all. The
        vacuum is what stops the images interacting, and every existing
        caller of this function wants it.

        ``"true"`` flags only the axes the structure genuinely
        repeats in -- a nanotube comes back ``(False, False, True)`` and a
        sheet ``(True, True, False)``. That is what the structure *is*,
        it is what every viewer and ASE-based tool reads correctly, and
        the Quantum ESPRESSO writer uses it to pick the k-mesh and
        ``assume_isolated``: a tube gets ``1 1 N`` rather than a mesh
        across its vacuum.

        Which to ask for is not cosmetic: the Quantum ESPRESSO writer
        reads ``pbc`` to choose the k-mesh and ``assume_isolated``, so a
        cage marked 3D is sampled 2x2x2 across its own vacuum where
        ``"true"`` gives it 1x1x1 and Makov-Payne, and a sheet 5x5x2
        where ``"true"`` gives 5x5x1 and the 2D correction.

    Returns
    -------
    ase.Atoms
        Periodic per ``mark``, with a ``unit_cell`` entry in
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

    if mark not in ("true", "3D"):
        raise ValueError(f"mark must be 'true' or '3D', not {mark!r}.")
    out.set_cell(cell)
    # Either way every axis now has a real lattice vector and the open
    # ones carry measured vacuum. The flag is about what the structure
    # claims to be, not about the geometry, which is the same.
    out.set_pbc(True if mark == "3D" else pbc)

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


def supercell(atoms: Atoms, counts: Sequence[int]) -> Atoms:
    """Repeat a periodic cell, carrying its bonds and ring census with it.

    ``Atoms.repeat`` copies the atoms and copies ``info`` verbatim, which
    for this package means a supercell that still claims the bond list,
    ring list and ring census of one cell. The preview then draws bonds on
    the first copy only, the Blender bundle exports a connectivity that
    covers an eighth of the structure, and the Euler check reports a sound
    structure as broken because the census no longer matches the atoms.
    None of that is visible in a bare atom count, which is what makes it
    worth a function.

    Bonds are re-indexed rather than re-guessed. Every bond of the original
    cell appears once per copy, and a bond that crossed a face now joins
    two copies — which copy is exactly what
    :func:`~nanocarbon_lab.utils.geometry.bond_shifts` already computes.
    The result is the same list ``guess_bonds`` would return on the
    supercell, at a fraction of the cost and without the risk of a
    different bond tolerance quietly changing the connectivity.

    Rings are carried the same way, by walking each ring from atom to atom
    and following the shift of each step, so a ring that straddles a face
    comes out whole. A ring that cannot be walked — which would mean the
    bond list and the ring list disagree — is dropped rather than guessed
    at, and the census is rebuilt from what survived.

    Parameters
    ----------
    atoms
        The structure to repeat.
    counts
        Copies along a, b and c. A count on a direction that is not
        periodic is an error, not something to silently obey: repeating
        vacuum stacks copies through each other.

    Returns
    -------
    ase.Atoms
        The supercell, with consistent ``info``.

    Raises
    ------
    ValueError
        If a count is below one, or asks to repeat an aperiodic direction.
    """
    from .utils.geometry import bond_shifts

    counts = tuple(int(n) for n in counts)
    if len(counts) != 3 or any(n < 1 for n in counts):
        raise ValueError(f"counts must be three integers >= 1, got {counts}")
    pbc = atoms.get_pbc()
    lengths = np.asarray(atoms.cell).astype(float)
    for index, (n, axis) in enumerate(zip(counts, "abc", strict=True)):
        if n == 1:
            continue
        if not pbc[index] or float(np.linalg.norm(lengths[index])) < 1e-6:
            raise ValueError(
                f"cannot repeat {n}x along {axis}: the structure is not "
                f"periodic there, so the copies would sit inside each other"
            )

    out = atoms.repeat(counts)
    total = counts[0] * counts[1] * counts[2]
    if total == 1:
        return out

    n_atoms = len(atoms)
    bonds = [list(map(int, pair)) for pair in atoms.info.get("bonds", [])]
    out.info = dict(atoms.info)
    out.info["supercell"] = list(counts)

    def index_of(cell_index: Sequence[int], atom: int) -> int:
        """ASE's own ordering: a slowest, c fastest, atoms within a copy."""
        i, j, k = cell_index
        copy = (i * counts[1] + j) * counts[2] + k
        return copy * n_atoms + atom

    images = [(i, j, k) for i in range(counts[0])
              for j in range(counts[1]) for k in range(counts[2])]

    if bonds:
        shifts = bond_shifts(atoms, bonds)
        # A bond drawn from atom i in copy M reaches atom j in copy M - s:
        # the shift is what has to be SUBTRACTED from j to bring it next
        # to i, so the partner lives one cell back along it.
        step = {}
        tiled = []
        for (first, second), shift in zip(bonds, shifts, strict=True):
            step[(first, second)] = tuple(int(v) for v in shift)
            step[(second, first)] = tuple(-int(v) for v in shift)
            for image in images:
                far = tuple((image[axis] - int(shift[axis])) % counts[axis]
                            for axis in range(3))
                tiled.append([index_of(image, first), index_of(far, second)])
        out.info["bonds"] = tiled
    else:
        step = {}

    rings = [list(map(int, ring)) for ring in atoms.info.get("rings", [])]
    carried = []
    for ring in rings:
        for image in images:
            walked = _walk_ring(ring, image, counts, step, index_of)
            if walked is None:
                carried = []
                break
            carried.append(walked)
        if rings and not carried:
            break
    if carried:
        out.info["rings"] = carried
        census: dict[int, int] = {}
        for ring in carried:
            census[len(ring)] = census.get(len(ring), 0) + 1
        out.info["ring_counts"] = census
    else:
        # Either there were no rings to carry, or the bond list cannot
        # account for one of them -- with no bonds recorded, for instance,
        # there is nothing to walk along. An incomplete ring list is worse
        # than none, because the colour-by-ring view would then be wrong
        # about which atoms are pentagons, so it is dropped. The census is
        # still exact: every ring appears once per copy.
        if rings:
            out.info.pop("rings", None)
        if "ring_counts" in atoms.info:
            out.info["ring_counts"] = {size: count * total for size, count
                                       in atoms.info["ring_counts"].items()}

    # The Euler budget is a sum over rings, so it scales with the number of
    # copies. Left alone, a sound 2x2x2 gyroid reports as BROKEN.
    components = int(atoms.info.get("n_shells", atoms.info.get("n_tubes", 1)))
    expected = atoms.info.get(
        "euler_expected",
        components * (12 - 12 * int(atoms.info.get("genus", 0))),
    )
    out.info["euler_expected"] = int(expected) * total
    for key in ("n_shells", "n_tubes"):
        if key in atoms.info:
            out.info[key] = int(atoms.info[key]) * total
    # Recorded once for the whole cell and no longer true of the supercell.
    for key in ("cell", "grid_resolution", "grid_retries"):
        out.info.pop(key, None)
    return out


def _walk_ring(ring, image, counts, step, index_of):
    """One ring of the original cell, placed in copy ``image``.

    Walked bond by bond so that a ring straddling a face keeps its far
    atoms in the neighbouring copy instead of folding back on itself.
    Returns ``None`` when a step is not in the bond list, which means the
    two records disagree and the ring should be dropped rather than
    invented.
    """
    here = list(image)
    walked = [index_of(here, ring[0])]
    for first, second in zip(ring, ring[1:], strict=False):
        shift = step.get((first, second))
        if shift is None:
            return None
        here = [(here[axis] - shift[axis]) % counts[axis] for axis in range(3)]
        walked.append(index_of(here, second))
    return walked


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
    "supercell",
    "to_unit_cell",
]
