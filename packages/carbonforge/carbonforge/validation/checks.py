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
from typing import Optional

import numpy as np
from ase import Atoms

from ..utils.constants import (
    HARD_MIN_DISTANCE,
    MAX_CC_DISTANCE,
    MIN_CC_DISTANCE,
)
from ..utils.geometry import minimum_image_distances
from ..topology.graph import coordination_numbers


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

    def merge(self, other: "ValidationReport") -> None:
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


def check_minimum_distances(atoms: Atoms) -> ValidationReport:
    """Flag any pair of atoms closer than :data:`HARD_MIN_DISTANCE`."""
    rep = ValidationReport()
    dmat = minimum_image_distances(atoms)
    np.fill_diagonal(dmat, np.inf)
    min_d = float(dmat.min())
    rep.info["min_interatomic_distance"] = min_d
    if min_d < HARD_MIN_DISTANCE:
        i, j = np.unravel_index(np.argmin(dmat), dmat.shape)
        rep.errors.append(
            f"Atoms {int(i)} and {int(j)} are only {min_d:.3f} Å apart "
            f"(< {HARD_MIN_DISTANCE} Å)."
        )
    elif min_d < MIN_CC_DISTANCE:
        rep.warnings.append(
            f"Shortest interatomic distance {min_d:.3f} Å is below the "
            f"typical sp2 range ({MIN_CC_DISTANCE}-{MAX_CC_DISTANCE} Å)."
        )
    return rep


def check_coordination(
    atoms: Atoms,
    tolerance: float = 0.30,
    allow_edge: bool = True,
) -> ValidationReport:
    """Check coordination numbers against sp2 carbon expectations.

    * coordination == 0 → isolated atom (error).
    * coordination >= 5 → unphysical for C/N/B/P/S in nanocarbons (error).
    * coordination == 1 → dangling atom (warning, error if ``not allow_edge``).
    * coordination == 2 → edge site (ok for ribbons/flakes, warning otherwise).
    * coordination == 3 or 4 → ok.
    """
    rep = ValidationReport()
    coord = coordination_numbers(atoms, tolerance=tolerance)
    n = len(atoms)
    rep.info["mean_coordination"] = float(coord.mean()) if n else 0.0

    n_iso = int(np.sum(coord == 0))
    n_one = int(np.sum(coord == 1))
    n_two = int(np.sum(coord == 2))
    n_high = int(np.sum(coord >= 5))

    if n_iso:
        rep.errors.append(f"{n_iso} isolated atom(s) (coordination 0).")
    if n_high:
        rep.errors.append(f"{n_high} atom(s) with coordination >= 5 (unphysical).")
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


def check_periodicity_coherence(
    atoms: Atoms,
    min_span_fraction: float = 0.05,
) -> ValidationReport:
    """Check that each periodic axis is one the structure actually extends along.

    Declaring an axis periodic when the atoms have no extent along it is a
    silent, total invalidation: the band path runs through vacuum, the k-mesh
    samples the empty direction while treating the real one as isolated, and
    the vacuum check inspects the wrong axes. Nothing crashes, and the input
    file looks perfectly well-formed.

    This is not hypothetical — the nanoribbon builder shipped with exactly
    that mistake, declaring the ribbon periodic along its vacuum direction,
    and every ribbon exported was wrong until it was caught.

    Choosing the criterion took some care, because the obvious ones fire on
    legitimate structures. A gap-based rule ("the atoms must nearly fill the
    cell") flags a disordered foam, whose random packing leaves real gaps of
    ~10 Å. An absolute span rule flags a two-atom primitive cell, whose atoms
    span only 0.7 Å along one lattice vector. What actually distinguishes the
    bug is that the atoms were **exactly coplanar** along the axis in
    question — zero extent, not merely small.

    So the test is a scale-free ratio: the atomic extent along a periodic
    axis must exceed ``min_span_fraction`` of that cell vector. Graphene's
    primitive cell sits at 0.33, the foam far higher, and the ribbon bug at
    0.00.

    Parameters
    ----------
    atoms
        Structure to check.
    min_span_fraction
        Minimum atomic extent along a periodic axis, as a fraction of the
        cell length along it.
    """
    report = ValidationReport()
    if len(atoms) < 2:
        return report

    positions = atoms.get_positions()
    cell = np.array(atoms.cell)
    for axis, periodic in enumerate(atoms.get_pbc()):
        if not periodic:
            continue
        span = float(positions[:, axis].max() - positions[:, axis].min())
        length = float(abs(cell[axis, axis]))
        report.info[f"span_axis_{axis}"] = span
        if length < 1e-6:
            report.errors.append(
                f"El eje {'xyz'[axis]} es periódico pero su vector de celda "
                "es nulo."
            )
            continue
        if span / length < min_span_fraction:
            name = "xyz"[axis]
            report.errors.append(
                f"El eje {name} está declarado periódico pero los átomos solo "
                f"se extienden {span:.2f} Å de los {length:.2f} Å de celda "
                f"({100 * span / length:.1f} %). Casi seguro que el eje "
                "periódico real es otro: revisa pbc. Con esto, el camino de "
                "bandas recorrería vacío y la malla k muestrearía la "
                "dirección equivocada, sin que nada dé error."
            )
    return report


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
    min_vacuum: Optional[float] = None,
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
    rep.merge(check_periodicity_coherence(atoms))
    rep.merge(check_minimum_distances(atoms))
    rep.merge(check_coordination(atoms, allow_edge=allow_edge))
    rep.merge(check_density(atoms))
    if not all(atoms.get_pbc()):
        rep.merge(check_vacuum(atoms, min_vacuum=min_vacuum or 10.0))
    return rep
