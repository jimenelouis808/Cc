"""Structural validation checks.

Each check returns a boolean (pass/fail) together with a human-readable
message, collected into a :class:`ValidationReport`. The philosophy is:

* **errors** are conditions that would make the simulation crash or be
  unphysical: atomic overlap, negative cell volume, missing vacuum on 2D
  slabs.
* **warnings** flag suspicious but not necessarily invalid geometries: an
  atom with coordination > 4, an unusually low/high density.

Downstream exporters should refuse to write a structure that has errors
unless explicitly forced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms

from ..topology.graph import coordination_numbers
from ..utils.constants import (
    COVALENT_RADII,
    DEFAULT_MAX_COORDINATION,
    HARD_MIN_DISTANCE,
    MAX_CC_DISTANCE,
    MAX_COORDINATION,
)


@dataclass
class ValidationReport:
    """Aggregate outcome of one or more validation checks."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True iff no error-level issue was recorded."""
        return len(self.errors) == 0

    def merge(self, other: ValidationReport) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.update(other.info)

    def summary(self) -> str:
        lines = [f"OK: {self.ok}"]
        if self.errors:
            lines.append("ERRORS:")
            lines.extend(f"  - {e}" for e in self.errors)
        if self.warnings:
            lines.append("WARNINGS:")
            lines.extend(f"  - {w}" for w in self.warnings)
        if self.info:
            lines.append("INFO:")
            lines.extend(f"  - {k}: {v}" for k, v in self.info.items())
        return "\n".join(lines)


#: Radius searched for the closest pair. Well past any bond in this
#: framework (the longest is a 2.76 Å Te-Te), so anything shorter is
#: found; a structure with nothing at all inside it is reported as such
#: rather than being searched exhaustively.
CLOSE_PAIR_CUTOFF: float = 4.0

#: Fraction of the two covalent radii below which a contact is shorter
#: than any real bond between those elements. A single bond sits at about
#: 1.00 of the sum, an sp2 C-C at 0.93 and a C=O double bond at 0.86, so
#: 0.80 leaves every genuine bond -- including the short ones a
#: functional group brings -- comfortably clear while still catching an
#: overlap that survived the hard minimum.
SHORT_BOND_FRACTION: float = 0.80


def check_minimum_distances(atoms: Atoms) -> ValidationReport:
    """Flag any pair of atoms closer than :data:`HARD_MIN_DISTANCE`.

    Uses a neighbour list rather than the full pairwise matrix. The
    matrix is O(N^2) in memory as well as time -- a twisted bilayer at
    the magic angle has 11 164 atoms and would want a gigabyte for it,
    and validation runs on the path of every export.
    """
    from ase.neighborlist import neighbor_list

    rep = ValidationReport()
    if len(atoms) < 2:
        rep.info["min_interatomic_distance"] = float("inf")
        return rep

    first, second, distance = neighbor_list(
        "ijd", atoms, cutoff=CLOSE_PAIR_CUTOFF)
    real = distance > 1e-9  # a periodic self-image sits at exactly zero
    if not real.any():
        # Nothing within the search radius: too sparse to have a close
        # contact, which is all this check is looking for.
        rep.info["min_interatomic_distance"] = float(CLOSE_PAIR_CUTOFF)
        return rep

    closest = int(np.argmin(np.where(real, distance, np.inf)))
    min_d = float(distance[closest])
    rep.info["min_interatomic_distance"] = min_d
    if min_d < HARD_MIN_DISTANCE:
        rep.errors.append(
            f"Atoms {int(first[closest])} and {int(second[closest])} are only "
            f"{min_d:.3f} Å apart (< {HARD_MIN_DISTANCE} Å)."
        )
        return rep

    # Short is only short *for the elements involved*. The sp2 window is
    # carbon's, and against it a perfectly ordinary O-H bond at 0.970 Å
    # -- the literature value exactly -- was reported as suspiciously
    # short on every hydroxylated structure. So the pair is judged
    # against its own covalent radii, the same element-aware principle
    # `MAX_COORDINATION` and `BOND_CUTOFF_OVERRIDE` already follow.
    symbols = atoms.get_chemical_symbols()
    radii = np.array([
        COVALENT_RADII.get(symbols[int(a)], 0.5 * MAX_CC_DISTANCE)
        + COVALENT_RADII.get(symbols[int(b)], 0.5 * MAX_CC_DISTANCE)
        for a, b in zip(first[real], second[real], strict=True)
    ])
    ratio = distance[real] / radii
    tightest = int(np.argmin(ratio))
    rep.info["shortest_bond_fraction"] = float(ratio[tightest])
    if ratio[tightest] < SHORT_BOND_FRACTION:
        left = symbols[int(first[real][tightest])]
        right = symbols[int(second[real][tightest])]
        rep.warnings.append(
            f"{left}-{right} pair at {float(distance[real][tightest]):.3f} Å is "
            f"{100 * ratio[tightest]:.0f}% of the sum of their covalent radii "
            f"({radii[tightest]:.2f} Å), shorter than any bond between them."
        )
    return rep


def check_coordination(
    atoms: Atoms,
    tolerance: float = 0.30,
    allow_edge: bool = True,
) -> ValidationReport:
    """Check coordination numbers against each element's own expectation.

    * coordination == 0 → isolated atom (error).
    * coordination above the element's maximum → unphysical (error).
    * coordination == 1 → dangling atom (warning, error if ``not allow_edge``).
    * coordination == 2 → edge site (ok for ribbons/flakes, warning otherwise).

    The ceiling is per element (:data:`MAX_COORDINATION`), not a flat 4.
    sp2 carbon never exceeds 4, but a dichalcogenide metal is
    six-coordinate by construction, and judging MoS2 against carbon's
    rule rejected every correct structure the ``tmd`` package builds.
    """
    rep = ValidationReport()
    coord = coordination_numbers(atoms, tolerance=tolerance)
    n = len(atoms)
    rep.info["mean_coordination"] = float(coord.mean()) if n else 0.0

    symbols = np.array(atoms.get_chemical_symbols())
    ceiling = np.array([
        MAX_COORDINATION.get(s, DEFAULT_MAX_COORDINATION) for s in symbols
    ])

    # A monovalent element at coordination 1 is *finished*, not dangling.
    # The ceiling here has always been per element; this test was not, so
    # every hydrogen of a grafted -OH and every fluorine of a fully
    # fluorinated sheet was reported as a defect -- 50 warnings on a
    # correct CF monolayer, which is the same mistake as judging a
    # six-coordinate metal by carbon's rule.
    monovalent = ceiling == 1

    # A builder may also declare atoms that are complete on one neighbour
    # for a reason coordination cannot see -- a carboxyl's carbonyl
    # oxygen is doubly bonded, so it has one neighbour and a full
    # valence. Without this every grafted carboxyl, carbonyl, nitro and
    # aldehyde read as a dangling defect.
    complete = monovalent.copy()
    for index in atoms.info.get("terminal_atoms", []):
        if 0 <= int(index) < n:
            complete[int(index)] = True

    n_iso = int(np.sum(coord == 0))
    n_one = int(np.sum((coord == 1) & ~complete))
    n_two = int(np.sum((coord == 2) & ~complete))
    over = coord > ceiling
    n_high = int(np.sum(over))

    if n_iso:
        rep.errors.append(f"{n_iso} isolated atom(s) (coordination 0).")
    if n_high:
        worst = symbols[over][0]
        rep.errors.append(
            f"{n_high} atom(s) over their element's coordination limit "
            f"(e.g. {worst} with {int(coord[over].max())}, max "
            f"{MAX_COORDINATION.get(worst, DEFAULT_MAX_COORDINATION)})."
        )
    if n_one:
        msg = f"{n_one} atom(s) with coordination 1 (dangling)."
        if allow_edge:
            rep.warnings.append(msg)
        else:
            rep.errors.append(msg)
    if n_two and not allow_edge:
        rep.warnings.append(f"{n_two} atom(s) with coordination 2 (edge).")
    return rep


def check_density(
    atoms: Atoms,
    min_density: float = 0.1,
    max_density: float = 3.5,
) -> ValidationReport:
    """Flag 3D densities outside a physically plausible window (g/cm³).

    Default window covers low-density aerogels (~0.1 g/cm³) up to graphite
    / diamond (~3.5 g/cm³). Only applied to fully periodic structures.
    """
    rep = ValidationReport()
    if not all(atoms.get_pbc()):
        rep.info["density_g_cm3"] = float("nan")
        return rep
    volume = float(abs(np.linalg.det(atoms.cell)))
    if volume <= 0:
        rep.errors.append("Non-positive cell volume.")
        return rep
    # Sum of atomic masses in u; 1 u/Å³ = 1.66054 g/cm³.
    mass = float(sum(atoms.get_masses()))
    density = mass / volume * 1.66054
    rep.info["density_g_cm3"] = density
    if density < min_density:
        rep.warnings.append(
            f"Density {density:.3f} g/cm³ below {min_density:.2f} g/cm³."
        )
    elif density > max_density:
        rep.warnings.append(
            f"Density {density:.3f} g/cm³ above {max_density:.2f} g/cm³."
        )
    return rep


def check_dimensionality(atoms: Atoms) -> ValidationReport:
    """Report the apparent dimensionality from the ``pbc`` flags."""
    rep = ValidationReport()
    dim = int(sum(atoms.get_pbc()))
    rep.info["dimensionality"] = dim
    return rep


def check_vacuum(
    atoms: Atoms,
    min_vacuum: float = 10.0,
) -> ValidationReport:
    """Ensure non-periodic directions have enough vacuum padding.

    Vacuum is computed as (cell length along axis) − (atomic span along axis).
    A non-periodic axis with < ``min_vacuum`` Å padding is flagged as error.
    """
    rep = ValidationReport()
    pbc = atoms.get_pbc()
    positions = atoms.get_positions()
    cell = np.array(atoms.cell)
    for ax in range(3):
        if pbc[ax]:
            continue
        length = float(cell[ax, ax])
        if length <= 0:
            rep.errors.append(f"Axis {ax}: non-positive cell length.")
            continue
        span = float(positions[:, ax].max() - positions[:, ax].min())
        vacuum = length - span
        rep.info[f"vacuum_axis_{ax}"] = vacuum
        if vacuum < min_vacuum:
            rep.errors.append(
                f"Axis {ax}: only {vacuum:.2f} Å of vacuum (need >= {min_vacuum} Å)."
            )
    return rep


def check_cell_consistency(atoms: Atoms) -> ValidationReport:
    """Check that the cell matrix is well defined and right-handed."""
    rep = ValidationReport()
    cell = np.array(atoms.cell)
    det = float(np.linalg.det(cell))
    rep.info["cell_volume"] = abs(det)
    if abs(det) < 1e-6:
        rep.errors.append("Degenerate cell (volume ≈ 0).")
    elif det < 0:
        rep.warnings.append("Left-handed cell (negative determinant).")
    return rep


def run_basic_checks(
    atoms: Atoms,
    allow_edge: bool = True,
    min_vacuum: float | None = None,
) -> ValidationReport:
    """Aggregate all basic checks.

    The ``min_vacuum`` threshold for :func:`check_vacuum` defaults to 10 Å
    for 1D/2D systems, and is skipped entirely for fully periodic (3D) cells.

    Parameters
    ----------
    atoms
        Structure to validate.
    allow_edge
        Passed through to :func:`check_coordination`. Set to ``False`` for
        strictly periodic bulk systems.
    min_vacuum
        Overrides the default vacuum threshold for non-periodic axes.
    """
    rep = ValidationReport()
    rep.merge(check_cell_consistency(atoms))
    rep.merge(check_dimensionality(atoms))
    rep.merge(check_minimum_distances(atoms))
    rep.merge(check_coordination(atoms, allow_edge=allow_edge))
    rep.merge(check_density(atoms))
    if not all(atoms.get_pbc()):
        rep.merge(check_vacuum(atoms, min_vacuum=min_vacuum or 10.0))
    return rep
