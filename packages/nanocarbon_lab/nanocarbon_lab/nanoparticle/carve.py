"""Cutting a nanoparticle out of an imported crystal.

The operation is one line -- keep the atoms whose signed distance is
negative -- and almost everything in this module is about the four things
that line does not decide and that change the answer:

**Where the centre is.** A sphere cut around a lattice point and the same
sphere cut around the hole between four of them are different particles:
different composition, different surface, and for a compound different
*charge*. Rock salt is the clearest case -- centred on a cation a cube
comes out cation-rich and centred on an anion, anion-rich -- and the
choice is not a detail, it is which of two structures you asked for. The
``centre`` argument makes it explicit and the report says what was
chosen. There is no neutral default; ``"cell"`` is merely the one that
needs no extra information.

**What the cut leaves hanging.** A plane through a crystal cuts bonds,
and some of what it leaves is attached by one bond or by none at all. A
singly-coordinated surface atom is not a facet, it is an artefact of the
cut: it has no counterpart on a real particle, it dominates any
subsequent relaxation, and it is the usual reason a carved particle
"explodes" in MD. They are trimmed by default, iteratively, because
removing one can leave its neighbour with one bond too.

**Stoichiometry, and therefore charge.** An arbitrary cut of a compound
is not stoichiometric. For MgO the excess is a few per cent and for a
layered or polar structure it can be much more, and in a plane-wave code
a charged cluster in a periodic box is a divergent sum that the code will
silently neutralise for you. The deviation is **always measured and
reported**; ``stoichiometric=True`` additionally trims the excess,
removing the most weakly bound atoms of the over-represented element
first. That is the standard practice and it is a heuristic, not a
minimisation: it removes the atoms that are least bound *by coordination
count*, which is not the same as least bound by energy. It is reported as
what it is.

**Whether a facet lands on a lattice plane.** This one is not a choice,
it is a trap. A cube of edge ``4a`` cut out of fcc gold around a lattice
point has its six faces lying *exactly* on planes of atoms, and what
those planes do then depends on the last bit of the arithmetic.
Measured, on the same 4a cube, changing the edge by **one part in a
million**:

    faces just inside the atom planes    171 atoms
    faces exactly on them                256 atoms
    faces just outside                   365 atoms

A factor of 2.1 between the first and the last, from a 1e-6 Å change in
the requested size. Nothing is wrong with any of the three -- they are
three different particles, and a real one is whichever the chemistry
terminated at -- but a size swept in steps of an Ångström walks through
all of them and reads as noise. Offsetting the cut so the faces fall
*between* the planes removes the tie, and then the count is exactly the
lattice's: 4n^3 for an n x n x n cube of fcc, for every n tried.

So :func:`carve_nanoparticle` counts the atoms sitting within
:data:`TIE_TOLERANCE` of the surface and says so in the report. It does
not move the cut: which plane terminates a facet is a statement about
the particle, and silently nudging it would be answering a chemical
question with a numerical convenience.

**What counts as the surface.** Not a guess: the bulk coordination of
each element is measured in the imported periodic cell, and a surface
atom is one with fewer neighbours than its own element has in the bulk.
So "surface fraction" means the same thing for a metal, an oxide and a
layered sulfide without any per-material thresholds, and a hollow
particle's inner wall counts as surface because it is one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs

from ..utils import DEFAULT_VACUUM_1D
from .crystal import supercell_for
from .shapes import Shape

#: Multiplies ASE's covalent-radius cutoffs when deciding what is bonded.
#: 1.2 is the usual slack for a crystal: it has to admit a stretched
#: surface bond and refuse a second-neighbour contact, and those are a
#: long way apart in every structure tried here.
BOND_SLACK = 1.2

#: An atom with this many neighbours or fewer is a leftover of the cut
#: rather than a surface site, and is trimmed.
LOOSE_COORDINATION = 1

#: How close to the cutting surface an atom has to be before the cut is
#: called commensurate with the lattice. 1e-3 Å is far below any physical
#: length and far above double precision on these coordinates, so an atom
#: inside it is on the plane rather than near it.
TIE_TOLERANCE = 1e-3

#: Composition is called stoichiometric within this relative deviation.
#: Not zero: a particle of a few hundred atoms cannot hit an exact ratio
#: for most cuts, and demanding it would trim a sound particle to
#: nothing.
STOICHIOMETRY_TOLERANCE = 0.02

CentreChoice = str


@dataclass
class CarveReport:
    """What came out of the cut, measured.

    Attributes
    ----------
    shape_label
        The shape's own description.
    n_atoms
        Atoms in the finished particle.
    composition
        Atoms per element.
    unit_ratio
        The imported cell's own element ratios, for comparison.
    stoichiometry_deviation
        Largest relative departure of the particle's ratios from the
        crystal's, as a fraction. ``0`` for an element.
    net_excess
        Atoms of each element in excess (positive) or deficit (negative)
        of the stoichiometric count, to the nearest atom.
    surface_atoms, surface_fraction
        How many atoms have fewer neighbours than their element does in
        the bulk, and what share of the particle that is.
    mean_coordination
        Averaged over all atoms.
    bulk_coordination
        Per element, measured in the periodic unit cell.
    min_distance
        Shortest interatomic distance in the particle, Å.
    max_radius
        Furthest atom from the centre, Å -- the particle's own half-size,
        which is not the shape's ``extent`` unless an atom happened to
        land on the corner.
    equivalent_diameter
        Diameter of the sphere of the same volume as the shape, Å. The
        one size that is comparable between a rod, a cube and a star, and
        the one a scattering or TEM size distribution is quoted as.
    shape_volume, atom_volume
        The shape's measured volume and the volume the atoms imply from
        the crystal's density. Their ratio is a check on the cut: well
        below 1 means the cut is losing material it should have kept.
    facets
        For a facetted shape, how many surface atoms sit on each face,
        keyed by the face normal. Empty for a round cut, which has no
        faces to count.
    atoms_on_the_cut
        How many atoms sit within :data:`TIE_TOLERANCE` of the cutting
        surface. Anything but zero means the facets are commensurate
        with the lattice and the atom count is sensitive to the last
        digit of the requested size; see the module docstring.
    trimmed_loose, trimmed_stoichiometry
        Atoms removed by each of the two trims.
    centre
        Which centring was used.
    hollow
        Whether the shape was a deliberately hollow one. A nanocage is
        almost all surface by construction, so the verdict has to tell it
        from a solid particle cut too small -- the two look identical in
        every number except the intent.
    warnings
        Everything worth saying before this particle is used.
    """

    shape_label: str
    n_atoms: int
    composition: Mapping[str, int]
    unit_ratio: Mapping[str, float]
    stoichiometry_deviation: float
    net_excess: Mapping[str, int]
    surface_atoms: int
    surface_fraction: float
    mean_coordination: float
    bulk_coordination: Mapping[str, float]
    min_distance: float
    max_radius: float
    equivalent_diameter: float
    shape_volume: float
    atom_volume: float
    facets: Mapping[str, int] = dataclass_field(default_factory=dict)
    atoms_on_the_cut: int = 0
    trimmed_loose: int = 0
    trimmed_stoichiometry: int = 0
    centre: str = "cell"
    hollow: bool = False
    warnings: list[str] = dataclass_field(default_factory=list)

    def summary(self) -> str:
        """The report as text, for the window and the command line."""
        formula = " ".join(f"{element}{count}"
                           for element, count in sorted(self.composition.items()))
        lines = [
            f"{self.shape_label}: {self.n_atoms} atoms, {formula}",
            f"  equivalent diameter {self.equivalent_diameter:.1f} A, "
            f"furthest atom {self.max_radius:.1f} A from the centre "
            f"(centred on {self.centre})",
            f"  surface {self.surface_atoms} atoms "
            f"({100 * self.surface_fraction:.1f} %), mean coordination "
            f"{self.mean_coordination:.2f}, shortest contact "
            f"{self.min_distance:.3f} A",
        ]
        if len(self.composition) > 1:
            excess = ", ".join(f"{element} {value:+d}"
                               for element, value in sorted(self.net_excess.items())
                               if value)
            lines.append(
                f"  stoichiometry off by "
                f"{100 * self.stoichiometry_deviation:.1f} %"
                + (f" ({excess})" if excess else " (exact)")
            )
        if self.facets:
            census = ", ".join(f"{face}: {count}"
                               for face, count in sorted(self.facets.items()))
            lines.append(f"  facet census: {census}")
        if self.trimmed_loose or self.trimmed_stoichiometry:
            lines.append(
                f"  trimmed {self.trimmed_loose} loose atoms"
                + (f" and {self.trimmed_stoichiometry} for stoichiometry"
                   if self.trimmed_stoichiometry else "")
            )
        lines += [f"  ! {text}" for text in self.warnings]
        return "\n".join(lines)


def carve_nanoparticle(
    crystal: Atoms,
    shape: Shape,
    centre: CentreChoice = "cell",
    trim_loose: bool = True,
    stoichiometric: bool = False,
    margin: float = 1.0,
    vacuum: float = DEFAULT_VACUUM_1D,
    max_atoms: int = 4_000_000,
) -> tuple[Atoms, CarveReport]:
    """Cut ``shape`` out of ``crystal`` and describe what came out.

    Parameters
    ----------
    crystal
        A periodic unit cell, as
        :func:`nanocarbon_lab.nanoparticle.crystal.read_crystal` returns.
    shape
        What to cut it to. Any
        :class:`~nanocarbon_lab.nanoparticle.shapes.Shape`.
    centre
        Where the shape's origin lands in the lattice:

        * ``"cell"`` -- on a corner of the unit cell, i.e. on a lattice
          point. Whatever sits at the cell origin ends up at the centre
          of the particle.
        * ``"void"`` -- on the body centre of the unit cell, which for
          most structures is an interstitial site.
        * ``"atom:X"`` -- on an atom of element ``X``, the first one in
          the cell. For a compound this is the control that decides
          whether the particle is cation- or anion-centred, and
          therefore its charge.

        None of these is more correct than the others; see the module
        docstring.
    trim_loose
        Remove atoms left with :data:`LOOSE_COORDINATION` neighbours or
        fewer, iteratively. On by default, and the one default here that
        is a judgement rather than a convenience: a singly-bonded atom
        is an artefact of the cut.
    stoichiometric
        Also trim the over-represented elements back to the crystal's own
        ratios, least-coordinated atoms first. Off by default because it
        *removes atoms the shape asked for*, and the caller should know
        that is happening.
    margin
        Extra Å of crystal built around the shape before cutting, so a
        facet landing exactly on a lattice plane is decided by the plane
        and not by the edge of the block.
    vacuum
        Padding around the finished particle in its cell, Å.
    max_atoms
        Guard on the block built before cutting; see
        :func:`~nanocarbon_lab.nanoparticle.crystal.supercell_for`.

    Returns
    -------
    (atoms, report)
        The particle, non-periodic, centred in a cell with ``vacuum``
        padding. ``atoms.info`` carries ``structure_type``,
        ``nanoparticle`` (the report as a dict), the shape's ``info`` and
        the source crystal's formula.

    Raises
    ------
    ValueError
        If the cut keeps no atoms -- a shape smaller than one unit cell,
        or a hollow shell thinner than one lattice spacing. Both are
        refused with the numbers, rather than returning an empty
        structure for something downstream to trip over.
    """
    if len(crystal) == 0:
        raise ValueError("the imported crystal has no atoms.")
    unit = crystal.copy()
    unit.pbc = True
    bulk_coordination = _bulk_coordination(unit)
    unit_ratio = _ratios(_composition(unit))

    shifted, centre_label = _apply_centring(unit, centre)
    half_widths = shape.bounding_box()
    block, block_info = supercell_for(shifted, half_widths, margin=margin,
                                      max_atoms=max_atoms)

    distance = shape(block.get_positions())
    inside = distance <= 0.0
    on_the_cut = int(np.count_nonzero(np.abs(distance) <= TIE_TOLERANCE))
    particle = block[inside]
    if len(particle) == 0:
        raise ValueError(
            f"nothing was kept: {shape.label} contains no lattice site. The "
            f"shape measures {np.round(2 * half_widths, 1).tolist()} A across "
            f"and the cell is "
            f"{np.round(np.linalg.norm(unit.cell.array, axis=1), 2).tolist()} "
            "A, so either the particle is smaller than one unit cell or a "
            "hollow wall is thinner than one lattice spacing."
        )

    warnings: list[str] = []
    trimmed_loose = 0
    if trim_loose:
        particle, trimmed_loose = _trim_loose(particle)
        if len(particle) == 0:
            raise ValueError(
                f"every atom in this cut was left with "
                f"{LOOSE_COORDINATION} bond(s) or fewer, so trimming emptied "
                "it. The particle is too small or too thin for this crystal."
            )

    trimmed_stoichiometry = 0
    if stoichiometric and len(unit_ratio) > 1:
        particle, trimmed_stoichiometry = _trim_to_stoichiometry(
            particle, unit_ratio)
        if trimmed_stoichiometry:
            warnings.append(
                f"stoichiometry was enforced by removing "
                f"{trimmed_stoichiometry} of the least-coordinated atoms of "
                "the over-represented element(s). That is the usual recipe "
                "and it is a heuristic: least bound by neighbour count is "
                "not least bound by energy."
            )
        if trim_loose:
            particle, extra = _trim_loose(particle)
            trimmed_loose += extra

    report = _measure(particle, shape, unit, unit_ratio, bulk_coordination,
                      centre_label, trimmed_loose, trimmed_stoichiometry,
                      on_the_cut, warnings)

    particle.info.update({
        "structure_type": "nanoparticle",
        "quality_family": None,
        "source_formula": unit.get_chemical_formula(mode="hill", empirical=True),
        "source_cell": np.asarray(unit.cell.array).tolist(),
        "shape": dict(shape.info),
        "shape_label": shape.label,
        "block": block_info,
        "nanoparticle": _report_as_dict(report),
    })
    _box_with_vacuum(particle, vacuum)
    return particle, report


def _box_with_vacuum(particle: Atoms, vacuum: float) -> None:
    """Give a finite particle an orthorhombic cell with padding.

    Sized from the atoms rather than from the shape: the shape's extent
    is reached by its corners and the atoms are at lattice sites, so the
    two differ by up to a lattice spacing and a cell built from the
    shape would leave the vacuum uneven. ``vacuum`` is the total padding
    per axis, which is this framework's convention everywhere else.
    """
    positions = particle.get_positions()
    span = positions.max(axis=0) - positions.min(axis=0)
    particle.set_cell(np.diag(span + float(vacuum)))
    particle.pbc = False
    particle.center()


# ----------------------------------------------------------------------
# centring
# ----------------------------------------------------------------------
def _apply_centring(unit: Atoms, centre: CentreChoice) -> tuple[Atoms, str]:
    """Shift the cell so the requested site sits at its origin."""
    shifted = unit.copy()
    if centre == "cell":
        return shifted, "cell"
    if centre == "void":
        offset = 0.5 * unit.cell.array.sum(axis=0)
        shifted.set_positions(shifted.get_positions() - offset)
        shifted.wrap()
        return shifted, "void (body centre of the cell)"
    if centre.startswith("atom:"):
        element = centre.split(":", 1)[1].strip()
        symbols = unit.get_chemical_symbols()
        if element not in symbols:
            raise ValueError(
                f"centre={centre!r} but the cell has no {element}: it has "
                f"{sorted(set(symbols))}."
            )
        index = symbols.index(element)
        offset = unit.get_positions()[index]
        shifted.set_positions(shifted.get_positions() - offset)
        shifted.wrap()
        return shifted, f"an {element} atom"
    raise ValueError(
        f"centre={centre!r} is not one of 'cell', 'void' or 'atom:<element>'."
    )


# ----------------------------------------------------------------------
# coordination
# ----------------------------------------------------------------------
def coordination(atoms: Atoms, slack: float = BOND_SLACK) -> np.ndarray:
    """Neighbour count per atom, from covalent radii.

    Uses ASE's ``natural_cutoffs`` scaled by ``slack``, and honours
    ``pbc``, so the same function gives the bulk reference on a periodic
    cell and the particle's own counts on a finite one. That identity is
    the point: a surface is defined here as *fewer neighbours than the
    bulk*, and the two numbers have to be measured the same way or the
    comparison means nothing.
    """
    if len(atoms) == 0:
        return np.zeros(0, dtype=int)
    cutoffs = [slack * value for value in natural_cutoffs(atoms)]
    neighbours = NeighborList(cutoffs, self_interaction=False, bothways=True,
                              skin=0.0)
    neighbours.update(atoms)
    return np.asarray([len(neighbours.get_neighbors(index)[0])
                       for index in range(len(atoms))], dtype=int)


def _bulk_coordination(unit: Atoms) -> dict[str, float]:
    """Mean coordination of each element in the periodic crystal."""
    counts = coordination(unit)
    out: dict[str, float] = {}
    symbols = np.asarray(unit.get_chemical_symbols())
    for element in sorted(set(symbols)):
        out[element] = float(counts[symbols == element].mean())
    return out


def _trim_loose(atoms: Atoms) -> tuple[Atoms, int]:
    """Iteratively drop atoms with too few neighbours.

    Iterative because the trim creates its own candidates: removing a
    dangling atom lowers its neighbour's count, and on a thin plate or a
    sharp apex that cascades for several rounds. It stops when a round
    removes nothing, which it always does -- each round strictly shrinks
    the structure.
    """
    removed = 0
    current = atoms
    while len(current):
        counts = coordination(current)
        keep = counts > LOOSE_COORDINATION
        if keep.all():
            break
        removed += int((~keep).sum())
        current = current[keep]
    return current, removed


# ----------------------------------------------------------------------
# stoichiometry
# ----------------------------------------------------------------------
def _composition(atoms: Atoms) -> dict[str, int]:
    out: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        out[symbol] = out.get(symbol, 0) + 1
    return out


def _ratios(composition: Mapping[str, int]) -> dict[str, float]:
    total = sum(composition.values())
    return {element: count / total for element, count in composition.items()}


def _stoichiometry_targets(composition: Mapping[str, int],
                           unit_ratio: Mapping[str, float]
                           ) -> dict[str, int]:
    """How many atoms of each element a stoichiometric particle has.

    Anchored on the element that is **least** over-represented relative
    to the crystal: that is the one that can be kept in full, and the
    others are trimmed down to match it. Anchoring on anything else
    would require adding atoms, which a cut cannot do.
    """
    limiting = min(composition[element] / ratio
                   for element, ratio in unit_ratio.items()
                   if element in composition and ratio > 0)
    return {element: int(math.floor(ratio * limiting + 1e-9))
            for element, ratio in unit_ratio.items()}


def _trim_to_stoichiometry(atoms: Atoms,
                           unit_ratio: Mapping[str, float]
                           ) -> tuple[Atoms, int]:
    """Remove the least-coordinated excess atoms until the ratios match."""
    composition = _composition(atoms)
    missing = [element for element in unit_ratio if element not in composition]
    if missing:
        raise ValueError(
            f"this cut contains no {', '.join(missing)} at all, so there is "
            "no ratio to trim towards. A particle missing one element of the "
            "compound is too small for it, or the centring put the cut "
            "between the sublattices."
        )
    targets = _stoichiometry_targets(composition, unit_ratio)
    counts = coordination(atoms)
    symbols = np.asarray(atoms.get_chemical_symbols())
    drop: list[int] = []
    for element, target in targets.items():
        excess = composition.get(element, 0) - target
        if excess <= 0:
            continue
        candidates = np.flatnonzero(symbols == element)
        # Least coordinated first; among equals, furthest from the centre,
        # because that is the one a real particle would lose.
        radius = np.linalg.norm(atoms.get_positions()[candidates], axis=1)
        order = np.lexsort((-radius, counts[candidates]))
        drop.extend(int(candidates[index]) for index in order[:excess])
    if not drop:
        return atoms, 0
    keep = np.ones(len(atoms), dtype=bool)
    keep[drop] = False
    return atoms[keep], len(drop)


# ----------------------------------------------------------------------
# measuring
# ----------------------------------------------------------------------
def _measure(particle: Atoms, shape: Shape, unit: Atoms,
             unit_ratio: Mapping[str, float],
             bulk_coordination: Mapping[str, float],
             centre_label: str, trimmed_loose: int,
             trimmed_stoichiometry: int, on_the_cut: int,
             warnings: list[str]) -> CarveReport:
    composition = _composition(particle)
    counts = coordination(particle)
    symbols = np.asarray(particle.get_chemical_symbols())
    surface = np.zeros(len(particle), dtype=bool)
    for element, bulk in bulk_coordination.items():
        mask = symbols == element
        # Strictly fewer than the bulk, with a small allowance so an
        # atom that merely sits in a slightly stretched site is not
        # called a surface atom.
        surface |= mask & (counts < bulk - 0.5)

    positions = particle.get_positions()
    radius = float(np.linalg.norm(positions, axis=1).max())
    minimum = _min_distance(positions)

    shape_volume = shape.volume()
    equivalent = 2.0 * (3.0 * shape_volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    cell_volume = float(abs(np.linalg.det(unit.cell.array)))
    atom_volume = len(particle) * cell_volume / len(unit)

    targets = (_stoichiometry_targets(composition, unit_ratio)
               if len(unit_ratio) > 1 and
               all(element in composition for element in unit_ratio) else {})
    excess = {element: composition.get(element, 0) - target
              for element, target in targets.items()}
    deviation = _deviation(composition, unit_ratio)

    if len(unit_ratio) > 1 and deviation > STOICHIOMETRY_TOLERANCE:
        warnings.append(
            f"composition is {100 * deviation:.1f} % off the crystal's "
            "ratios. For an ionic compound that means a charged cluster, "
            "which a plane-wave code will neutralise behind your back; pass "
            "stoichiometric=True, or change `centre`, or accept it knowingly."
        )
    if surface.sum() > 0.6 * len(particle) and not shape.info.get("hollow"):
        warnings.append(
            f"{100 * surface.sum() / len(particle):.0f} % of the atoms are on "
            "the surface. That is a real property of a small particle rather "
            "than an error, but nothing about this structure will behave like "
            "the bulk material."
        )
    if on_the_cut:
        warnings.append(
            f"{on_the_cut} atoms lie exactly on the cutting surface, so this "
            "cut is commensurate with the lattice and the atom count is "
            "decided by the last digit of the size you asked for -- a 4a cube "
            "of fcc gold reads 171, 256 or 365 atoms across a 1e-6 A change. "
            "Offset the size by a fraction of the interplanar spacing to put "
            "the facets between the atom planes, or keep this termination "
            "deliberately."
        )
    if minimum < 0.8:
        warnings.append(
            f"shortest contact is {minimum:.3f} A. The cut does not move "
            "atoms, so this came from the imported cell -- check it for "
            "partial occupancies expanded onto one site."
        )
    if atom_volume < 0.5 * shape_volume:
        warnings.append(
            f"the atoms fill {100 * atom_volume / shape_volume:.0f} % of the "
            "shape's volume. A hollow or spiky shape should read low here; a "
            "solid one should not, and if it does the shape is smaller than "
            "the lattice can resolve."
        )

    return CarveReport(
        shape_label=shape.label,
        n_atoms=len(particle),
        composition=composition,
        unit_ratio=dict(unit_ratio),
        stoichiometry_deviation=deviation,
        net_excess=excess,
        surface_atoms=int(surface.sum()),
        surface_fraction=float(surface.sum() / len(particle)),
        mean_coordination=float(counts.mean()),
        bulk_coordination=dict(bulk_coordination),
        min_distance=minimum,
        max_radius=radius,
        equivalent_diameter=equivalent,
        shape_volume=shape_volume,
        atom_volume=atom_volume,
        facets=_facet_census(particle, shape, surface),
        atoms_on_the_cut=on_the_cut,
        trimmed_loose=trimmed_loose,
        trimmed_stoichiometry=trimmed_stoichiometry,
        centre=centre_label,
        hollow=bool(shape.info.get("hollow", False)),
        warnings=warnings,
    )


def _deviation(composition: Mapping[str, int],
               unit_ratio: Mapping[str, float]) -> float:
    """Largest relative departure from the crystal's element ratios."""
    if len(unit_ratio) <= 1:
        return 0.0
    total = sum(composition.values())
    worst = 0.0
    for element, ratio in unit_ratio.items():
        have = composition.get(element, 0) / total
        worst = max(worst, abs(have - ratio) / ratio)
    return float(worst)


def _min_distance(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return math.inf
    from scipy.spatial import cKDTree
    distances, _ = cKDTree(positions).query(positions, k=2)
    return float(distances[:, 1].min())


def _facet_census(particle: Atoms, shape: Shape,
                  surface: np.ndarray) -> dict[str, int]:
    """How many surface atoms sit on each face of a facetted shape.

    Only meaningful for a shape built from half-spaces, which is why it
    reads the normals out of the shape's own ``info`` instead of trying
    to perceive facets from the atom positions. A round cut returns an
    empty census rather than inventing faces for it: a sphere has no
    facets, and a census that listed some would be describing the
    sampling, not the particle.
    """
    normals = shape.info.get("normals")
    if not normals or not surface.any():
        return {}
    stack = np.asarray(normals, dtype=float)
    distances = np.asarray(shape.info["distances"], dtype=float)
    positions = particle.get_positions()[surface]
    # Which face each surface atom is closest to, in its own plane's units.
    gaps = distances[None, :] - positions @ stack.T
    nearest = np.argmin(np.abs(gaps), axis=1)
    census: dict[str, int] = {}
    for index in nearest:
        key = _normal_key(stack[index])
        census[key] = census.get(key, 0) + 1
    return census


def _normal_key(normal: Sequence[float]) -> str:
    """A face normal as a short readable label."""
    return "(" + " ".join(f"{value:+.2f}" for value in normal) + ")"


def _report_as_dict(report: CarveReport) -> dict[str, Any]:
    """The report as plain types, so it survives into a file."""
    return {
        "shape_label": report.shape_label,
        "n_atoms": report.n_atoms,
        "composition": dict(report.composition),
        "stoichiometry_deviation": report.stoichiometry_deviation,
        "net_excess": dict(report.net_excess),
        "surface_atoms": report.surface_atoms,
        "surface_fraction": report.surface_fraction,
        "mean_coordination": report.mean_coordination,
        "bulk_coordination": dict(report.bulk_coordination),
        "min_distance": report.min_distance,
        "max_radius": report.max_radius,
        "equivalent_diameter": report.equivalent_diameter,
        "shape_volume": report.shape_volume,
        "atom_volume": report.atom_volume,
        "facets": dict(report.facets),
        "atoms_on_the_cut": report.atoms_on_the_cut,
        "trimmed_loose": report.trimmed_loose,
        "trimmed_stoichiometry": report.trimmed_stoichiometry,
        "centre": report.centre,
        "hollow": report.hollow,
        "warnings": list(report.warnings),
    }


__all__ = [
    "BOND_SLACK",
    "LOOSE_COORDINATION",
    "STOICHIOMETRY_TOLERANCE",
    "TIE_TOLERANCE",
    "CarveReport",
    "carve_nanoparticle",
    "coordination",
]
