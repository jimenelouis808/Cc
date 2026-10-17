"""Reading a crystal in, and turning Miller indices into real normals.

Everything a shape needs to know about the material it is going to cut,
and nothing it does not. Three jobs:

**Read the file and say what came in.** :func:`read_crystal` accepts a
``.cif`` and everything else ASE reads, and returns the cell with a
report of what it found: composition, density, the shortest contact, and
whether the file described a periodic crystal at all. A ``.cif`` holding
a molecule with no cell is a perfectly valid file and a useless input to
a carver, so it is refused by name rather than carved into a
nanoparticle of one unit.

**Repeat it until the shape fits inside.** :func:`supercell_for` works
out how many unit cells each axis needs and builds them. The arithmetic
is the part worth writing down once: a shape reaching ``extent`` in every
direction needs the supercell to cover ``2 * extent`` along each
*Cartesian* axis, and a unit cell vector does not generally lie along
one, so the count per axis follows from the cell's **inverse**, not from
the lengths of its vectors. Using ``extent / |a_i|`` is right for a cubic
cell and short for a monoclinic one, which is the kind of error that
shows up as a particle with two flat sides.

**Give the facets their normals.** A Miller plane ``(hkl)`` has normal
``h b1 + k b2 + l b3`` in the *reciprocal* lattice -- not ``(h, k, l)``
in Cartesian coordinates, which they only coincide with for a cubic
cell. :func:`miller_normal` does that conversion and
:func:`miller_family` finds the symmetry-equivalent set.

**On the symmetry used for a family.** The equivalent set comes from the
*lattice's* point group, found by brute force over integer matrices that
preserve the metric tensor -- no spglib, and exact for every Bravais
lattice in its conventional setting, because the generators of those
groups all have entries in ``{-1, 0, 1}`` there. What that gives is the
**holohedry**: the symmetry of the lattice, which may be higher than the
symmetry of the crystal sitting on it. For rock-salt MgO the two agree
and {100} is six equivalent faces. For a structure whose basis breaks
the lattice symmetry -- any ordered alloy, wurtzite's polar axis, a
Jahn-Teller distortion -- they do not, and a "family" then contains
faces that are crystallographically different and may well have
different surface energies. The report says which group was found and
how large it is, so that assumption is visible rather than buried; when
it is wrong, pass the normals you want explicitly.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase import io as ase_io

#: Below this the file is not describing a periodic crystal.
MIN_CELL_LENGTH = 0.5

#: Integer entries searched when looking for the lattice's point group.
#: ``{-1, 0, 1}`` is enough for every Bravais lattice in its conventional
#: setting: the hexagonal 6-fold, the cubic 3-fold and every mirror are
#: all written with those entries there. A non-conventional (reduced,
#: supercell, or arbitrarily transformed) cell can need wider entries and
#: will come back with a smaller group than it really has -- which costs
#: facets in a family, never adds wrong ones.
POINT_GROUP_ENTRIES = (-1, 0, 1)


@dataclass
class CrystalReport:
    """What the imported file turned out to contain.

    Attributes
    ----------
    source
        The path it was read from.
    formula
        Reduced chemical formula of the unit cell.
    counts
        Atoms per element in the unit cell.
    cell_lengths, cell_angles
        a, b, c in Å and alpha, beta, gamma in degrees.
    volume
        Unit-cell volume, Å^3.
    density
        g/cm^3, from the cell's own mass and volume.
    min_distance
        Shortest interatomic distance under the minimum image, Å. A
        value well under 1 Å usually means the file has partial
        occupancies expanded onto the same site, which is the one
        failure mode of a downloaded ``.cif`` that produces a
        nanoparticle full of overlapping atoms rather than an error.
    point_group_order
        Size of the lattice point group that was found.
    lattice_system
        The system that order implies, as a name.
    warnings
        Everything worth saying out loud before the cell is used.
    """

    source: Path
    formula: str
    counts: Mapping[str, int]
    cell_lengths: tuple[float, float, float]
    cell_angles: tuple[float, float, float]
    volume: float
    density: float
    min_distance: float
    point_group_order: int
    lattice_system: str
    warnings: list[str] = dataclass_field(default_factory=list)

    def summary(self) -> str:
        """One paragraph, for the window and the command line."""
        a, b, c = self.cell_lengths
        alpha, beta, gamma = self.cell_angles
        lines = [
            f"{self.source.name}: {self.formula}, "
            f"{sum(self.counts.values())} atoms in the cell",
            f"  a={a:.4f} b={b:.4f} c={c:.4f} A, "
            f"alpha={alpha:.2f} beta={beta:.2f} gamma={gamma:.2f} deg",
            f"  V={self.volume:.2f} A^3, rho={self.density:.3f} g/cm^3, "
            f"shortest contact {self.min_distance:.3f} A",
            f"  lattice point group: order {self.point_group_order} "
            f"({self.lattice_system})",
        ]
        lines += [f"  ! {text}" for text in self.warnings]
        return "\n".join(lines)


def read_crystal(path: str | Path, index: int = -1,
                 ) -> tuple[Atoms, CrystalReport]:
    """Read a crystal structure and describe it.

    Parameters
    ----------
    path
        A ``.cif``, or any other format ASE can read -- the extension is
        a hint and the content decides, the same rule the rest of this
        framework follows.
    index
        Which frame, for a file holding several. The default takes the
        last, which is the relaxed one in an optimisation trajectory.

    Returns
    -------
    (atoms, report)
        The unit cell as read, with ``pbc`` forced on in the directions
        that have a cell vector, and a :class:`CrystalReport`.

    Raises
    ------
    ValueError
        If the file has no usable cell. A carver needs a lattice to
        repeat; a molecule is not one, and cutting a sphere out of a
        single molecule would return that molecule and call it a
        nanoparticle.
    """
    path = Path(path)
    atoms = ase_io.read(str(path), index=index)
    if isinstance(atoms, list):                 # pragma: no cover - ase API
        atoms = atoms[-1]
    cell = np.asarray(atoms.cell.array, dtype=float)
    lengths = np.linalg.norm(cell, axis=1)
    if np.any(lengths < MIN_CELL_LENGTH) or abs(np.linalg.det(cell)) < 1e-6:
        raise ValueError(
            f"{path.name} has no usable unit cell (lengths "
            f"{np.round(lengths, 3).tolist()} A). A nanoparticle is cut out "
            "of a repeated lattice, so the file has to describe a periodic "
            "crystal -- if this is a molecule, there is nothing to repeat."
        )
    atoms = atoms.copy()
    atoms.pbc = True

    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    volume = float(abs(np.linalg.det(cell)))
    # 1 amu / A^3 = 1.66053906660 g/cm^3
    density = float(atoms.get_masses().sum()) / volume * 1.66053906660
    minimum = _minimum_contact(atoms)
    group = lattice_point_group(cell)
    system = lattice_system(len(group))

    warnings: list[str] = []
    if minimum < 0.8:
        warnings.append(
            f"shortest contact is {minimum:.3f} A, which is shorter than any "
            "real bond. A downloaded .cif with partial occupancies expands "
            "to several atoms on one site; pick one before carving, or every "
            "particle cut from this cell will carry the overlap."
        )
    if len(group) <= 2:
        warnings.append(
            "the lattice point group came out of order "
            f"{len(group)}, which means this cell is triclinic or is not in "
            "a conventional setting. Miller families will contain only the "
            "faces that symmetry justifies, so a facetted habit built from "
            "them will have fewer faces than its name suggests."
        )
    return atoms, CrystalReport(
        source=path,
        formula=atoms.get_chemical_formula(mode="hill", empirical=True),
        counts=counts,
        cell_lengths=tuple(float(value) for value in lengths),  # type: ignore
        cell_angles=tuple(float(value)
                          for value in atoms.cell.angles()),  # type: ignore
        volume=volume,
        density=density,
        min_distance=minimum,
        point_group_order=len(group),
        lattice_system=system,
        warnings=warnings,
    )


def _minimum_contact(atoms: Atoms) -> float:
    """Shortest interatomic distance under the minimum image, in Å.

    Includes an atom's own periodic images, which is the point: a cell
    whose single atom sits 0.4 Å from its own image across a face is a
    broken cell, and a pair-distance check inside the cell alone would
    call it perfect.
    """
    if len(atoms) == 0:                         # pragma: no cover - empty file
        return math.inf
    repeated = atoms.repeat((2, 2, 2))
    positions = repeated.get_positions()
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    distances, _ = tree.query(positions, k=2)
    return float(distances[:, 1].min())


# ----------------------------------------------------------------------
# lattice symmetry, without spglib
# ----------------------------------------------------------------------
def lattice_point_group(cell: np.ndarray,
                        tolerance: float = 1e-4) -> list[np.ndarray]:
    """Integer matrices that leave the lattice's metric unchanged.

    An operation maps the lattice onto itself exactly when it sends
    every lattice vector to another lattice vector, so in the basis of
    the cell it is an **integer** matrix ``U``; and it is a rotation or
    reflection exactly when it preserves all lengths and angles, i.e.
    when ``U^T G U = G`` for the metric tensor ``G = A A^T``. Both
    conditions are exact statements, so the group can be enumerated by
    searching integer matrices rather than approximated -- which is what
    makes this honest without a symmetry library.

    The search runs over entries in :data:`POINT_GROUP_ENTRIES`, which
    covers every Bravais lattice in its conventional setting. The result
    is the lattice's holohedry and may be larger than the crystal's own
    point group; see the module docstring.

    Returns
    -------
    list of (3, 3) integer arrays
        Always contains the identity, so it is never empty.
    """
    matrix = np.asarray(cell, dtype=float).reshape(3, 3)
    metric = matrix @ matrix.T
    scale = float(np.abs(metric).max())
    candidates = _candidate_matrices()
    # `U^T G U == G` for every candidate at once. Written as one einsum
    # because the loop over 19683 matrices took 0.3 s, and a family of
    # facets asks for this group once per family.
    transformed = np.einsum("nij,jk,nlk->nil", candidates, metric, candidates)
    keep = np.all(np.abs(transformed - metric) <= tolerance * scale,
                  axis=(1, 2))
    return [candidates[index].astype(int) for index in np.flatnonzero(keep)]


@lru_cache(maxsize=1)
def _candidate_matrices() -> np.ndarray:
    """Every integer 3x3 matrix over :data:`POINT_GROUP_ENTRIES` whose
    determinant is +-1 -- i.e. every candidate lattice automorphism."""
    entries = np.asarray(list(itertools.product(POINT_GROUP_ENTRIES,
                                                repeat=9)), dtype=float)
    stack = entries.reshape(-1, 3, 3)
    determinant = np.rint(np.linalg.det(stack))
    return np.ascontiguousarray(stack[np.abs(determinant) == 1])


#: Order of the holohedral point group of each lattice system. The map is
#: one-way on purpose: an order identifies the system, but a system does
#: not identify a cell, and this is used only to put a readable name next
#: to the number in a report.
HOLOHEDRY_ORDER = {
    2: "triclinic",
    4: "monoclinic",
    8: "orthorhombic",
    16: "tetragonal",
    12: "trigonal (rhombohedral)",
    24: "hexagonal",
    48: "cubic",
}


def lattice_system(order: int) -> str:
    """Name the lattice system a point-group order implies."""
    return HOLOHEDRY_ORDER.get(order, f"unrecognised (order {order})")


def reciprocal(cell: np.ndarray) -> np.ndarray:
    """Reciprocal lattice vectors as rows, without the 2 pi.

    ``B = (A^-1)^T``, so ``b_i . a_j = delta_ij``. The 2 pi is left out
    because the only use here is plane normals, where it cancels; a
    diffraction calculation would want it and is not this module's job.
    """
    matrix = np.asarray(cell, dtype=float).reshape(3, 3)
    return np.linalg.inv(matrix).T


def miller_normal(cell: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Unit normal of the ``(hkl)`` plane, in Cartesian coordinates.

    ``n = h b1 + k b2 + l b3``, normalised. For a cubic cell that comes
    out parallel to ``(h, k, l)`` and for anything else it does not: in
    a calcite-like rhombohedral cell (a = 6.375 Å, alpha = 46.1 deg) the
    (104) normal -- the cleavage plane, the one every calcite rhomb is
    bounded by -- comes out **15.5 degrees** away from the direction its
    indices would point if they were read as Cartesian components. A
    15-degree error in a facet normal is a different facet.
    """
    h, k, ell = (float(value) for value in indices)
    if h == k == ell == 0:
        raise ValueError("(000) is not a plane.")
    vector = np.asarray([h, k, ell], dtype=float) @ reciprocal(cell)
    return vector / float(np.linalg.norm(vector))


def miller_family(cell: np.ndarray, indices: Sequence[int],
                  tolerance: float = 1e-4) -> np.ndarray:
    """Every normal symmetry-equivalent to ``(hkl)``, as unit vectors.

    The orbit of the index triple under the lattice point group, taken
    in **index space** -- an operation ``U`` acting on direct-lattice
    coordinates acts on Miller indices as ``(hkl) U``, because indices
    are covariant -- and then converted to Cartesian normals.

    Both ``(hkl)`` and ``(-h-k-l)`` are included. They are the same
    *plane* but opposite *faces*, and a particle has both; leaving one
    out gives a solid cut on one side only.
    """
    group = lattice_point_group(cell, tolerance=tolerance)
    triple = np.asarray([int(round(float(value))) for value in indices])
    orbit: set[tuple[int, int, int]] = set()
    for operation in group:
        mapped = triple @ operation
        orbit.add(tuple(int(value) for value in mapped))
    normals = np.asarray([miller_normal(cell, item) for item in sorted(orbit)])
    # Two different index triples can give the same direction once
    # normalised (e.g. (1,1,1) and (2,2,2) if both were somehow in the
    # orbit). Deduplicate on the direction, which is what a face is.
    keep: list[np.ndarray] = []
    for normal in normals:
        if not any(float(normal @ other) > 1.0 - 1e-6 for other in keep):
            keep.append(normal)
    return np.asarray(keep)


def family_normals_for(cell: np.ndarray):
    """A ``normals_for`` callable bound to this cell, for :func:`wulff`.

    :func:`nanocarbon_lab.nanoparticle.shapes.wulff` needs a map from a
    family key to its normals, and without one it falls back to the
    cubic families. This supplies the imported crystal's own, so a Wulff
    construction on a hexagonal or rhombohedral cell is right rather
    than cubic-shaped.
    """
    from .shapes import _parse_indices

    def normals_for(key: str) -> np.ndarray:
        return miller_family(cell, _parse_indices(key))

    return normals_for


def interplanar_spacing(cell: np.ndarray, indices: Sequence[int]) -> float:
    """``d_hkl`` in Å -- the spacing of the ``(hkl)`` planes.

    Useful next to a face distance: a facet distance that is not a whole
    number of ``d_hkl`` from the centre cuts the last plane in half, and
    what that leaves depends on the basis. Reported rather than
    enforced; see
    :func:`nanocarbon_lab.nanoparticle.carve.carve_nanoparticle`.
    """
    h, k, ell = (float(value) for value in indices)
    vector = np.asarray([h, k, ell], dtype=float) @ reciprocal(cell)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("(000) has no spacing.")
    return 1.0 / norm


# ----------------------------------------------------------------------
# building the block the shape is cut out of
# ----------------------------------------------------------------------
def repeats_for(cell: np.ndarray, half_widths: Sequence[float],
                margin: float = 1.0) -> tuple[int, int, int]:
    """How many unit cells each axis needs to cover a box.

    Parameters
    ----------
    cell
        Unit cell, rows as vectors.
    half_widths
        How far the shape reaches along +-x, +-y, +-z in Å.
    margin
        Extra Å added on every side, so a plane that lands exactly on a
        layer of atoms is decided by the cut rather than by the edge of
        the block.

    Notes
    -----
    The count along cell vector ``i`` is **not** ``extent / |a_i|``. The
    shape's bounding box is axis-aligned in Cartesian space, and a
    general cell vector is oblique to it; what is needed is how many
    cells it takes to reach the box's corners, which is a question about
    the *inverse* cell. Taking the fractional coordinates of all eight
    corners and spanning them is exact and is what this does. For a
    cubic cell it reduces to the naive formula; for calcite's
    rhombohedral cell the naive formula is 27 % short along every axis.
    """
    matrix = np.asarray(cell, dtype=float).reshape(3, 3)
    half = np.asarray(half_widths, dtype=float) + float(margin)
    if np.any(~np.isfinite(half)):
        raise ValueError(
            "the shape has no finite extent, so there is no block to build. "
            "An unbounded shape (a bare half-space) has to be intersected "
            "with something closed first."
        )
    corners = np.asarray([[sx * half[0], sy * half[1], sz * half[2]]
                          for sx in (1, -1) for sy in (1, -1)
                          for sz in (1, -1)])
    fractional = corners @ np.linalg.inv(matrix)
    span = fractional.max(axis=0) - fractional.min(axis=0)
    # `ceil(span) + 1`: `span` cells cover the extent only if the block
    # happens to start in phase with it, and it does not in general.
    return tuple(int(math.ceil(value)) + 1 for value in span)  # type: ignore


def supercell_for(atoms: Atoms, half_widths: Sequence[float],
                  margin: float = 1.0,
                  max_atoms: int = 4_000_000) -> tuple[Atoms, dict[str, Any]]:
    """Repeat ``atoms`` until it covers the box, centred on the origin.

    Returns the block with its centre of mass of the *lattice* -- the
    centre of the repeated cell, not the centroid of the atoms -- moved
    to the origin, because that is the point every shape is built
    around.

    ``max_atoms`` guards the one mistake that is expensive rather than
    wrong: a 500 Å particle of a large cell is tens of millions of
    atoms, and building it to then throw 99 % away takes all the memory
    in the machine. The limit refuses with the number it would have
    built, so the caller can decide.
    """
    repeats = repeats_for(atoms.cell.array, half_widths, margin=margin)
    total = len(atoms) * repeats[0] * repeats[1] * repeats[2]
    if total > max_atoms:
        raise ValueError(
            f"covering this shape needs {repeats} unit cells, which is "
            f"{total:,} atoms before anything is cut away (the limit is "
            f"{max_atoms:,}). Either the particle is very large or the "
            "imported cell is: check the cell is primitive, or raise "
            "max_atoms deliberately."
        )
    block = atoms.repeat(repeats)
    centre = 0.5 * np.asarray(repeats, dtype=float) @ atoms.cell.array
    block.set_positions(block.get_positions() - centre)
    block.pbc = False
    block.set_cell(None)
    return block, {"repeats": list(repeats), "block_atoms": len(block),
                   "unit_cell_atoms": len(atoms)}


__all__ = [
    "CrystalReport",
    "HOLOHEDRY_ORDER",
    "MIN_CELL_LENGTH",
    "POINT_GROUP_ENTRIES",
    "family_normals_for",
    "interplanar_spacing",
    "lattice_point_group",
    "lattice_system",
    "miller_family",
    "miller_normal",
    "read_crystal",
    "reciprocal",
    "repeats_for",
    "supercell_for",
]
