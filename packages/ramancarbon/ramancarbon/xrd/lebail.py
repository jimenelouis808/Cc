"""Whole-pattern fitting without a structure: Le Bail and Pawley.

Rietveld needs the atoms. Often they are not known — a new phase, a
distorted one, or simply a cell you want to refine before committing to a
structural model — and then the intensities can be treated as free numbers
instead of being calculated. What is left is still enough to refine the
cell, the zero, the background and the peak shape, which is most of what
a diffraction pattern is asked for.

Two ways to do it, both here because they fail differently:

**Le Bail** starts every reflection at the same intensity and then, at
each cycle, hands each reflection the share of the observed counts that
its own profile claims. It is stable, it never returns a negative
intensity, and for reflections that overlap exactly it splits them evenly
— which is not a measurement, it is the starting guess surviving to the
end.

**Pawley** makes the intensities genuine parameters and solves for them.
Since the pattern is linear in them, that is one non-negative least
squares per cycle, not an iteration. It gives the intensities their own
uncertainties, and on a heavily overlapped pattern it is where the
correlation becomes visible instead of being hidden by Le Bail's
arbitrary split.

**Neither gives structure factors you can hand to a structure solver
without looking at the overlap.** For a well-separated reflection the
extracted intensity is a measurement; for a pair 0.01° apart it is a
convention. The overlap is computed and reported per reflection so that
distinction is visible rather than assumed.

And a whole-pattern fit **always** beats a Rietveld fit of the same
pattern, because it has one free number per reflection. Comparing their
R factors says nothing about the structure. What a Le Bail R_wp is good
for is a floor: it is the best any structural model of this cell could
do, and a Rietveld fit far above it has a structural problem, while one
close to it has none left to find.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares, nnls

from .pattern import Pattern
from .powder import Profile, pseudo_voigt, reflections
from .rietveld import (
    PROFILE_CUTOFF,
    RefinementError,
    _angle_correction,
    chebyshev_background,
    r_factors,
)
from .structure import Crystal

#: Reflections closer than this fraction of a FWHM are treated as one
#: measurement, not two.
OVERLAP_FRACTION = 0.5


@dataclass
class ExtractedLine:
    """One reflection with the intensity the pattern gave it."""

    hkl: tuple[int, int, int]
    two_theta: float
    d: float
    intensity: float
    uncertainty: Optional[float] = None
    overlap: float = 0.0
    """How much of this reflection's profile is shared with its
    neighbours, 0 to 1. Above ~0.5 the intensity is a split, not a
    measurement."""

    def __str__(self) -> str:
        mark = " (solapada)" if self.overlap > OVERLAP_FRACTION else ""
        return (f"{self.hkl}  2θ={self.two_theta:7.3f}°  d={self.d:7.4f} Å  "
                f"I={self.intensity:10.1f}{mark}")


@dataclass
class ExtractionResult:
    """A whole-pattern fit with no structural model."""

    method: str
    pattern: Pattern
    calculated: np.ndarray
    background: np.ndarray
    lines: list[ExtractedLine]
    lattice_factors: tuple[float, float, float] = (1.0, 1.0, 1.0)
    profile: Profile = field(default_factory=Profile)
    zero: float = 0.0
    displacement: float = 0.0
    r_p: float = 0.0
    r_wp: float = 0.0
    r_expected: float = 0.0
    gof: float = 0.0
    cycles: int = 0
    converged: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def difference(self) -> np.ndarray:
        return self.pattern.intensity - self.calculated

    def refined_crystal(self, crystal: Crystal) -> Crystal:
        """The input structure with the refined cell."""
        if self.lattice_factors == (1.0, 1.0, 1.0):
            return crystal
        return crystal.with_lattice(crystal.lattice.scaled(self.lattice_factors))

    def overlapped(self) -> list[ExtractedLine]:
        return [line for line in self.lines if line.overlap > OVERLAP_FRACTION]

    def describe(self) -> str:
        overlapped = len(self.overlapped())
        return (f"{self.method}: R_wp = {100 * self.r_wp:.2f} %, "
                f"GoF = {self.gof:.2f}, {len(self.lines)} reflexiones "
                f"({overlapped} solapadas), {self.cycles} ciclos")


def _lines_of(crystal: Crystal, wavelength: float, window: tuple[float, float],
              factors: tuple[float, float, float]):
    """Allowed reflections of a cell, positions only."""
    scaled = (crystal if factors == (1.0, 1.0, 1.0)
              else crystal.with_lattice(crystal.lattice.scaled(factors)))
    return reflections(scaled, wavelength=wavelength, two_theta_range=window,
                       normalise=False, min_relative=0.0)


def _centres(crystal: Crystal, hkl, wavelength: float,
             factors: tuple[float, float, float], zero: float,
             displacement: float = 0.0) -> np.ndarray:
    """Where a fixed set of reflections sits for a given cell.

    The hkl list is enumerated ONCE, from the starting cell, and every
    later position is computed from it. Re-enumerating each cycle looks
    equivalent and is not: a reflection crossing the edge of the window
    changes the length of the intensity vector mid-refinement, and the
    intensities then belong to different reflections than the ones they
    were extracted for.
    """
    lattice = (crystal.lattice if factors == (1.0, 1.0, 1.0)
               else crystal.lattice.scaled(factors))
    d = np.asarray(lattice.d_spacing(list(hkl)), dtype=float)
    ratio = np.clip(wavelength / (2.0 * np.maximum(d, 1e-9)), -1.0, 1.0)
    angles = 2.0 * np.degrees(np.arcsin(ratio))
    return angles + _angle_correction(angles, zero, displacement)


def _profile_matrix(
    angles: np.ndarray,
    centres: np.ndarray,
    profile: Profile,
    satellites: Optional[np.ndarray] = None,
    ratio: float = 0.0,
) -> np.ndarray:
    """``(n_points, n_reflections)`` of area-normalised peak shapes.

    Building it explicitly is what makes Pawley one linear solve and Le
    Bail one matrix product per cycle, instead of a loop over reflections
    inside a loop over cycles.

    The Kα₂ satellite goes into the SAME column as its parent, at the
    fixed intensity ratio: it is the same reflection seen in the other
    line of the doublet, not a second free intensity. Leaving it out
    entirely — the obvious simplification — is not neutral: the fit has to
    cover the satellite with something, and what it uses is the peak
    width, which came out four times too large.
    """
    matrix = np.zeros((angles.size, centres.size))
    widths = profile.fwhm(centres)
    mixings = profile.eta(centres)
    for index, (centre, width, mixing) in enumerate(zip(centres, widths, mixings)):
        window = ((angles > centre - PROFILE_CUTOFF * width)
                  & (angles < centre + PROFILE_CUTOFF * width))
        if window.any():
            matrix[window, index] = pseudo_voigt(angles[window], centre,
                                                 width, mixing)
        if satellites is None or ratio <= 0.0:
            continue
        second = float(satellites[index])
        window = ((angles > second - PROFILE_CUTOFF * width)
                  & (angles < second + PROFILE_CUTOFF * width))
        if window.any():
            matrix[window, index] += ratio * pseudo_voigt(
                angles[window], second, width, mixing)
    return matrix


def _overlaps(centres: np.ndarray, profile: Profile) -> np.ndarray:
    """Per reflection, how close its nearest neighbour is in FWHM units."""
    if centres.size < 2:
        return np.zeros(centres.size)
    widths = profile.fwhm(centres)
    order = np.argsort(centres)
    sorted_centres = centres[order]
    gaps = np.full(centres.size, np.inf)
    distances = np.diff(sorted_centres)
    gaps[order[:-1]] = np.minimum(gaps[order[:-1]], distances)
    gaps[order[1:]] = np.minimum(gaps[order[1:]], distances)
    return np.clip(1.0 - gaps / np.maximum(widths, 1e-9), 0.0, 1.0)


def le_bail(
    pattern: Pattern,
    crystal: Crystal,
    profile: Optional[Profile] = None,
    background_order: int = 6,
    cycles: int = 20,
    refine_cell: bool = True,
    refine_profile: bool = True,
    warmup: int = 1,
    partitions: int = 25,
    tolerance: float = 1e-5,
) -> ExtractionResult:
    """Extract intensities and refine the cell, without a structure.

    The intensities start equal and are re-partitioned every cycle in
    proportion to what each reflection's own profile contributes at each
    point — Le Bail's original recipe, and the reason it cannot produce a
    negative intensity.

    Parameters
    ----------
    crystal:
        Provides the cell and the **symmetry**: which reflections are
        allowed is a property of the space group, and a Le Bail fit that
        includes systematically absent reflections will happily give them
        intensity and improve its R factor by fitting noise.
    refine_cell, refine_profile:
        Whether to let the cell and the Caglioti coefficients move between
        cycles. Both on by default, which is what the method is for.
    warmup:
        Cycles of pure partitioning before the cell and the shape are
        allowed to move. Staged, for the same reason the Rietveld
        refinement here is staged.
    partitions:
        Partition passes per cycle. See the comment in the loop: the
        partition converges slowly and costs nothing, and refining a cell
        against half-converged intensities is what sends the peak width to
        four times its true value.
    """
    angles = np.asarray(pattern.two_theta, dtype=float)
    observed = np.asarray(pattern.intensity, dtype=float)
    weights = 1.0 / np.maximum(np.abs(observed), 1.0)
    shape = profile or _seed_profile(pattern)
    window = (float(angles[0]) - 1.0, float(angles[-1]) + 1.0)
    doublet = _doublet(pattern)

    lines = _lines_of(crystal, pattern.wavelength, window, (1.0, 1.0, 1.0))
    if not lines:
        raise RefinementError(
            "esta celda no produce ninguna reflexión permitida en el "
            f"intervalo {angles[0]:.1f}–{angles[-1]:.1f}°"
        )
    partitions = max(1, int(partitions))

    hkl = [line.hkl for line in lines]
    factors = [1.0, 1.0, 1.0]
    constraint = crystal.lattice_constraint()
    zero = 0.0
    # The starting intensities are scaled so the model's total AREA is the
    # observed peak area, not so each peak is as tall as the tallest one.
    # Starting from max(observed)/n makes the first model enormous, the
    # first background fit strongly negative to compensate, and the two
    # never recover — the fit settles a per cent off in the cell with the
    # peak widths inflated to cover the damage.
    background = _initial_background(angles, observed, background_order)
    peak_area = float(np.sum(np.maximum(
        observed - chebyshev_background(angles, background), 0.0))
        * float(np.median(np.diff(angles))))
    intensities = np.full(len(lines), max(peak_area, 1.0) / len(lines))
    previous = np.inf
    converged = False
    used = 0

    for cycle in range(cycles):
        used = cycle + 1
        centres = _centres(crystal, hkl, pattern.wavelength, tuple(factors), zero)
        matrix = _profile_matrix(angles, centres, shape,
                                 *_satellites(crystal, hkl, doublet,
                                              tuple(factors), zero))

        # Le Bail's partition: the observed counts at each point are shared
        # out among the reflections in proportion to what each one's own
        # profile contributes there, and a reflection's new intensity is
        # the sum of its shares.
        #
        #   I_k ← I_k · Σᵢ Pₖᵢ (yᵢ − bᵢ)/Σⱼ Iⱼ Pⱼᵢ  ⁄  Σᵢ Pₖᵢ
        #
        # The division by Σᵢ Pₖᵢ is what makes it a fixed point: with the
        # model already right, every intensity comes back unchanged.
        #
        # It is run MANY times per cycle, and that is not a detail. From
        # uniform starting intensities the partition improves R_wp by
        # about a tenth of itself per pass — 24 % after ten passes, 7 %
        # after forty — and a cycle that partitions once and then refines
        # is refining a cell and a peak width against intensities that are
        # still wrong by a factor of two. Each pass is two matrix products
        # and costs about a millisecond; the refinement it feeds costs two
        # hundred times that.
        column_sums = np.maximum(matrix.sum(axis=0), 1e-12)
        for _ in range(partitions):
            model = matrix @ intensities
            background = _fit_background(angles, observed - model,
                                         background_order)
            residual = observed - chebyshev_background(angles, background)
            floor = 1e-6 * float(np.max(model)) if model.size else 1e-12
            with np.errstate(invalid="ignore", divide="ignore"):
                share = np.where(model > floor,
                                 residual / np.maximum(model, floor), 0.0)
            intensities = np.maximum(
                intensities * (matrix.T @ share) / column_sums, 0.0)

        calculated = matrix @ intensities + chebyshev_background(angles, background)
        current = float(np.sum(weights * (observed - calculated) ** 2))

        if cycle >= warmup and (refine_cell or refine_profile):
            factors, zero, shape = _refine_nonlinear(
                angles, observed, weights, crystal, hkl, pattern.wavelength,
                intensities, background, factors, zero, shape, constraint,
                refine_cell, refine_profile, doublet,
            )

        if (cycle > warmup
                and abs(previous - current) <= tolerance * max(previous, 1.0)):
            converged = True
            break
        previous = current

    return _finish("Le Bail", pattern, crystal, hkl, lines, intensities,
                   shape, tuple(factors), zero, background, used, converged,
                   uncertainties=None)


def pawley(
    pattern: Pattern,
    crystal: Crystal,
    profile: Optional[Profile] = None,
    background_order: int = 6,
    cycles: int = 12,
    refine_cell: bool = True,
    refine_profile: bool = True,
    tolerance: float = 1e-5,
) -> ExtractionResult:
    """The same fit with the intensities as genuine parameters.

    The pattern is linear in the intensities once the positions and the
    shape are fixed, so each cycle is one non-negative least squares
    rather than an iteration. Non-negativity is imposed rather than hoped
    for: an unconstrained solve puts negative intensity on one member of
    an overlapped pair and a compensating excess on the other, fits
    beautifully, and means nothing.
    """
    angles = np.asarray(pattern.two_theta, dtype=float)
    observed = np.asarray(pattern.intensity, dtype=float)
    weights = 1.0 / np.maximum(np.abs(observed), 1.0)
    shape = profile or _seed_profile(pattern)
    window = (float(angles[0]) - 1.0, float(angles[-1]) + 1.0)
    doublet = _doublet(pattern)

    lines = _lines_of(crystal, pattern.wavelength, window, (1.0, 1.0, 1.0))
    if not lines:
        raise RefinementError(
            "esta celda no produce ninguna reflexión permitida en el intervalo")

    hkl = [line.hkl for line in lines]
    factors = [1.0, 1.0, 1.0]
    constraint = crystal.lattice_constraint()
    zero = 0.0
    background = np.zeros(background_order)
    intensities = np.zeros(len(lines))
    matrix = np.zeros((angles.size, len(lines)))
    previous = np.inf
    converged = False
    used = 0

    for cycle in range(cycles):
        used = cycle + 1
        centres = _centres(crystal, hkl, pattern.wavelength, tuple(factors), zero)
        matrix = _profile_matrix(angles, centres, shape,
                                 *_satellites(crystal, hkl, doublet,
                                              tuple(factors), zero))

        # Solve for the intensities AND the background together: fitting
        # them in alternation lets each absorb the other's error and the
        # pair never settles.
        design = np.concatenate(
            [matrix, _chebyshev_matrix(angles, background_order)], axis=1)
        root = np.sqrt(weights)
        solution, _ = nnls(design * root[:, None], observed * root,
                           maxiter=50 * design.shape[1])
        intensities = solution[:matrix.shape[1]]
        background = solution[matrix.shape[1]:]

        calculated = design @ solution
        current = float(np.sum(weights * (observed - calculated) ** 2))

        if refine_cell or refine_profile:
            factors, zero, shape = _refine_nonlinear(
                angles, observed, weights, crystal, hkl, pattern.wavelength,
                intensities, background, factors, zero, shape, constraint,
                refine_cell, refine_profile, doublet,
            )

        if cycle and abs(previous - current) <= tolerance * max(previous, 1.0):
            converged = True
            break
        previous = current

    uncertainties = _intensity_errors(angles, observed, weights, matrix,
                                      intensities, background_order)
    return _finish("Pawley", pattern, crystal, hkl, lines, intensities,
                   shape, tuple(factors), zero, background, used, converged,
                   uncertainties=uncertainties)


def _chebyshev_matrix(angles: np.ndarray, order: int) -> np.ndarray:
    """The background basis, so it can be solved for linearly."""
    columns = []
    for index in range(order):
        coefficients = [0.0] * order
        coefficients[index] = 1.0
        columns.append(chebyshev_background(angles, coefficients))
    return np.stack(columns, axis=1) if columns else np.zeros((angles.size, 0))


def _initial_background(angles: np.ndarray, observed: np.ndarray,
                        order: int) -> np.ndarray:
    """A Chebyshev through the pattern's lower envelope.

    Fitted to the running minimum rather than to the data, because a fit
    to the data is a fit to the peaks. It only has to be close: the cycles
    refine it.
    """
    if order <= 0:
        return np.zeros(0)
    window = max(5, observed.size // 40)
    padded = np.pad(observed, window, mode="edge")
    envelope = np.asarray([
        float(np.min(padded[index:index + 2 * window + 1]))
        for index in range(observed.size)
    ])
    return _fit_background(angles, envelope, order)


def _fit_background(angles: np.ndarray, residual: np.ndarray,
                    order: int) -> np.ndarray:
    """Least-squares Chebyshev background of a residual."""
    if order <= 0:
        return np.zeros(0)
    design = _chebyshev_matrix(angles, order)
    solution, *_ = np.linalg.lstsq(design, residual, rcond=None)
    return solution


def _seed_profile(pattern: Pattern) -> Profile:
    """A starting peak shape measured off the pattern.

    The default Caglioti coefficients describe somebody else's
    diffractometer. Starting from them is what makes a Le Bail fit crawl:
    the intensity partition cannot get below about 35 % R_wp while the
    modelled peaks are half the width of the real ones, so by the time the
    shape is released the intensities are still wrong, and the two chase
    each other into a wide-peak minimum. Two seconds of peak finding fixes
    it, and it is what a person does by eye before starting.
    """
    from .search import find_peaks

    try:
        peaks = [p.fwhm for p in find_peaks(pattern) if p.fwhm]
    except Exception:                                # noqa: BLE001
        peaks = []
    if not peaks:
        return Profile()
    width = float(np.median(peaks))
    return Profile(u=0.0, v=0.0, w=max(width ** 2, 1e-5), eta0=0.5)


def _doublet(pattern: Pattern) -> tuple[float, float]:
    """``(wavelength of Kα₂, its intensity ratio)``, or a zero ratio."""
    if not getattr(pattern, "has_doublet", False):
        return (pattern.wavelength, 0.0)
    return (pattern.kalpha2_lambda, float(pattern.kalpha2_ratio))


def _satellites(crystal, hkl, doublet, factors, zero):
    """The Kα₂ centres and ratio to pass to :func:`_profile_matrix`."""
    second, ratio = doublet
    if ratio <= 0.0:
        return (None, 0.0)
    return (_centres(crystal, hkl, second, factors, zero), ratio)


def _refine_nonlinear(
    angles, observed, weights, crystal, hkl, wavelength,
    intensities, background, factors, zero, shape, constraint,
    refine_cell, refine_profile, doublet=None,
):
    """Move the cell, the zero and the peak shape, intensities held."""
    # `constraint` is one LABEL per axis; equal labels refine together, so
    # a hexagonal cell gets one parameter for a and b, not two.
    free_labels = sorted(set(constraint)) if refine_cell else []
    start: list[float] = [
        factors[constraint.index(label)] for label in free_labels
    ]
    if refine_cell:
        start.append(zero)
    if refine_profile:
        start += [shape.u, shape.v, shape.w, shape.eta0]
    if not start:
        return factors, zero, shape

    def unpack(values):
        cursor = 0
        new_factors = list(factors)
        if refine_cell:
            for label in free_labels:
                for target, source in enumerate(constraint):
                    if source == label:
                        new_factors[target] = float(values[cursor])
                cursor += 1
            new_zero = float(values[cursor])
            cursor += 1
        else:
            new_zero = zero
        new_shape = shape
        if refine_profile:
            new_shape = Profile(
                u=float(values[cursor]), v=float(values[cursor + 1]),
                w=float(max(values[cursor + 2], 1e-5)),
                eta0=float(np.clip(values[cursor + 3], 0.0, 1.0)),
            )
        return new_factors, new_zero, new_shape

    background_curve = chebyshev_background(angles, background)
    root = np.sqrt(weights)

    def residuals(values):
        new_factors, new_zero, new_shape = unpack(values)
        centres = _centres(crystal, hkl, wavelength, tuple(new_factors), new_zero)
        matrix = _profile_matrix(
            angles, centres, new_shape,
            *_satellites(crystal, hkl, doublet or (wavelength, 0.0),
                         tuple(new_factors), new_zero))
        return root * (observed - (matrix @ intensities + background_curve))

    # Bounded, and not by taste: with an unbounded solver the width runs
    # to its floor on one cycle and comes back out at twice the right
    # value on the next, and the fit settles there.
    lower, upper = _bounds(start, refine_cell, len(free_labels), refine_profile)
    outcome = least_squares(residuals, start, bounds=(lower, upper),
                            method="trf", max_nfev=150)
    return unpack(outcome.x)


def _bounds(start, refine_cell, axes, refine_profile):
    """Physical limits on the cell scale, the zero and the Caglioti terms."""
    lower: list[float] = []
    upper: list[float] = []
    if refine_cell:
        lower += [0.9] * axes + [-1.0]
        upper += [1.1] * axes + [1.0]
    if refine_profile:
        lower += [-0.1, -0.1, 1e-4, 0.0]
        upper += [1.0, 1.0, 4.0, 1.0]
    for index, value in enumerate(start):
        lower[index] = min(lower[index], value - 1e-9)
        upper[index] = max(upper[index], value + 1e-9)
    return lower, upper


def _intensity_errors(angles, observed, weights, matrix, intensities, order):
    """Standard errors of the Pawley intensities from the normal matrix."""
    design = np.concatenate([matrix, _chebyshev_matrix(angles, order)], axis=1)
    scale = np.sqrt(weights)[:, None]
    scaled = design * scale
    try:
        covariance = np.linalg.pinv(scaled.T @ scaled)
    except np.linalg.LinAlgError:                    # pragma: no cover
        return None
    model = design @ np.concatenate([intensities, np.zeros(order)])
    residual = np.sqrt(weights) * (observed - model)
    dof = max(angles.size - design.shape[1], 1)
    variance = float(np.sum(residual ** 2)) / dof
    return np.sqrt(np.maximum(np.diag(covariance)[:matrix.shape[1]], 0.0) * variance)


def _finish(method, pattern, crystal, hkl, lines, intensities, shape, factors,
            zero, background, cycles, converged, uncertainties):
    angles = np.asarray(pattern.two_theta, dtype=float)
    centres = _centres(crystal, hkl, pattern.wavelength, factors, zero)
    matrix = _profile_matrix(angles, centres, shape,
                             *_satellites(crystal, hkl, _doublet(pattern),
                                          factors, zero))
    background_curve = chebyshev_background(angles, background)
    calculated = matrix @ intensities + background_curve

    lattice = (crystal.lattice if factors == (1.0, 1.0, 1.0)
               else crystal.lattice.scaled(factors))
    spacings = np.asarray(lattice.d_spacing(list(hkl)), dtype=float)
    overlap = _overlaps(centres, shape)
    extracted = [
        ExtractedLine(
            hkl=tuple(indices), two_theta=float(centre), d=float(spacing),
            intensity=float(value),
            uncertainty=(float(uncertainties[index])
                         if uncertainties is not None else None),
            overlap=float(overlap[index]),
        )
        for index, (indices, centre, spacing, value) in enumerate(
            zip(hkl, centres, spacings, intensities))
    ]

    weights = 1.0 / np.maximum(np.abs(pattern.intensity), 1.0)
    r_p, r_wp, r_expected, gof = r_factors(
        pattern.intensity, calculated, weights,
        free=len(intensities) + len(background))

    warnings = [
        "un ajuste de patrón completo SIEMPRE ajusta mejor que un Rietveld "
        "de lo mismo, porque tiene un número libre por reflexión: comparar "
        "sus factores R no dice nada de la estructura. Lo que da es un "
        "suelo, lo mejor que podría hacer cualquier modelo estructural de "
        "esta celda"
    ]
    overlapped = sum(1 for line in extracted if line.overlap > OVERLAP_FRACTION)
    if overlapped:
        warnings.append(
            f"{overlapped} de {len(extracted)} reflexiones se solapan más de "
            "media anchura: para esas, la intensidad extraída es un reparto, "
            "no una medida, y no debe pasarse a un programa de resolución "
            "estructural como si lo fuera"
        )
    if method == "Le Bail":
        warnings.append(
            "Le Bail reparte a partes iguales las reflexiones que coinciden "
            "exactamente: ese reparto es la suposición inicial sobreviviendo "
            "hasta el final, no un resultado del ajuste"
        )
    if not converged:
        warnings.append(
            f"no ha convergido en {cycles} ciclos: los números están sin "
            "asentar")
    empty = int(np.count_nonzero(np.asarray(intensities) <= 0))
    if empty > 0.2 * max(len(intensities), 1):
        warnings.append(
            f"{empty} reflexiones han quedado en cero: o la celda no es la de "
            "esta muestra, o el grupo espacial permite reflexiones que en esta "
            "estructura están extinguidas"
        )

    return ExtractionResult(
        method=method, pattern=pattern, calculated=calculated,
        background=background_curve, lines=extracted,
        lattice_factors=factors, profile=shape, zero=zero,
        r_p=r_p, r_wp=r_wp, r_expected=r_expected, gof=gof,
        cycles=cycles, converged=converged, warnings=warnings,
    )


def cell_from_extraction(result: ExtractionResult, crystal: Crystal) -> dict:
    """The refined cell, in the units it is quoted in."""
    refined = result.refined_crystal(crystal)
    lattice = refined.lattice
    return {
        "a": lattice.a, "b": lattice.b, "c": lattice.c,
        "alpha": lattice.alpha, "beta": lattice.beta, "gamma": lattice.gamma,
        "volumen": lattice.volume,
        "cambio_relativo": tuple(f - 1.0 for f in result.lattice_factors),
    }


def figure_of_merit(result: ExtractionResult) -> float:
    """De Wolff's M20-like index: how well the cell explains the lines.

    Uses the mean absolute discrepancy between where the refined cell puts
    a reflection and where a peak actually is. Larger is better; below
    about 10 the cell is not established, whatever the R factor says —
    a wrong cell with enough reflections can fit any pattern.
    """
    observed = result.pattern
    if len(result.lines) < 2:
        return 0.0
    positions = np.asarray([line.two_theta for line in result.lines])
    inside = positions[(positions >= observed.two_theta[0])
                       & (positions <= observed.two_theta[-1])]
    if inside.size < 2:
        return 0.0
    spacing = float(np.median(np.diff(np.sort(inside))))
    width = float(np.median(result.profile.fwhm(inside)))
    return float(spacing / max(width, 1e-6)) * math.sqrt(inside.size)


__all__ = [
    "OVERLAP_FRACTION",
    "ExtractedLine",
    "ExtractionResult",
    "cell_from_extraction",
    "figure_of_merit",
    "le_bail",
    "pawley",
]
