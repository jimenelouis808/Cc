"""How much of a phase could be present and still not show.

"Not detected" and "not there" are different statements, and only the
first is ever a measurement. A phase whose strongest reflection would sit
under the noise is invisible at any abundance below some limit, and the
useful answer is that limit -- not silence, and not a refined weight
fraction of 0.03 % that reads like a measurement of almost nothing when
it is really the fit shrugging.

The calculation is the detection threshold run backwards. For the phase's
strongest reflection inside the measured range:

1. the peak it would make is spread over the width the pattern's other
   reflections show, so the counts land over `span` points;
2. it is detectable when `height * sqrt(span) / sigma` clears the same
   threshold the peak search uses -- matched-filter significance, so a
   broad low peak and a narrow tall one are judged alike;
3. that fixes the largest invisible height, and the phase's scale
   follows from it;
4. the scale becomes a weight fraction through the same Hill-Howard
   product the refinement uses, against the phases that ARE present.

The number that comes out is an upper limit at the confidence the
threshold encodes, and it should be quoted as one: "below 1.8 %", never
"1.8 %".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionLimit:
    """The most of a phase that could hide in a pattern."""

    phase: str
    weight_fraction: float | None
    """Upper limit as a fraction of the identified crystalline part, or
    ``None`` when the phase has no reflection inside the measured range
    or no atomic weight to normalise with."""
    two_theta: float | None
    """Where the strongest reflection would have been."""
    hkl: tuple[int, int, int] | None
    reason: str

    def __str__(self) -> str:
        if self.weight_fraction is None:
            return f"{self.phase}: no se puede acotar — {self.reason}"
        return (f"{self.phase}: por debajo del "
                f"{100.0 * self.weight_fraction:.2g} % "
                f"(su reflexión más fuerte {self.hkl} caería en "
                f"{self.two_theta:.2f}° y no se ve)")


def detection_limit(pattern, crystal, present, threshold: float,
                    instrument_fwhm: float = 0.06) -> DetectionLimit:
    """Upper limit on `crystal`, given the `present` phases as reference.

    Parameters
    ----------
    pattern:
        The measured diffractogram, un-subtracted.
    crystal:
        The phase to bound.
    present:
        The refined :class:`PhaseModel` objects that ARE in the sample.
        They set the scale-to-weight conversion; without them a scale is
        a number with no meaning.
    threshold:
        Significance a peak must clear to be found, from the same search
        that produced the peak list.
    """
    from .powder import reflections
    from .rietveld import PhaseModel, calculate_pattern

    lines = reflections(crystal, wavelength=pattern.wavelength,
                        two_theta_range=pattern.range)
    if not lines:
        return DetectionLimit(crystal.name, None, None, None,
                              "ninguna reflexión cae dentro del barrido")
    strongest = max(lines, key=lambda r: r.intensity)
    if strongest.intensity <= 0:
        return DetectionLimit(crystal.name, None, None, None,
                              "su reflexión más fuerte tiene intensidad nula")

    two_theta = np.asarray(pattern.two_theta, dtype=float)
    index = int(np.argmin(np.abs(two_theta - strongest.two_theta)))
    sigma = pattern.sigma
    noise = (float(np.asarray(sigma)[index]) if sigma is not None
             else float(pattern.noise_estimate()))
    noise = max(noise, 1e-9)

    # The width it would have: whatever the phases that ARE there show.
    # Using the instrument width alone makes the limit far too tight on a
    # nanocrystalline sample, where every real reflection is several
    # times broader and therefore much harder to see.
    widths = [w for w in (_typical_fwhm(phase) for phase in present) if w]
    fwhm = max(widths) if widths else instrument_fwhm
    span = max(fwhm / max(pattern.step, 1e-9), 1.0)

    # The tallest peak the search would still have missed.
    max_height = threshold * noise / math.sqrt(span)

    # Convert that height into this phase's scale through the SAME
    # calculation the refinement uses. Going via the reflection's
    # relative intensity instead -- a 0-100 number -- mixes two
    # normalisations, and the first version of this function did exactly
    # that and returned an upper limit of 100 % for every phase, which
    # is true and useless.
    probe = PhaseModel(crystal=crystal, scale=1.0)
    calculated = calculate_pattern(two_theta, [probe], pattern.wavelength)
    reference = float(np.max(calculated))
    if not np.isfinite(reference) or reference <= 0:
        return DetectionLimit(crystal.name, None, strongest.two_theta,
                              strongest.hkl,
                              "el patrón calculado de esta fase es nulo")
    max_scale = max_height / reference

    products = []
    for phase in present:
        mass = getattr(phase, "cell_mass", None)
        if mass is None:
            return DetectionLimit(crystal.name, None, strongest.two_theta,
                                  strongest.hkl,
                                  "una fase presente no tiene masa tabulada")
        volume = phase.current_crystal().lattice.volume
        products.append(max(phase.scale, 0.0) * mass * volume)
    own_mass = _cell_mass(crystal)
    if own_mass is None:
        return DetectionLimit(crystal.name, None, strongest.two_theta,
                              strongest.hkl,
                              "no hay masa tabulada para esta fase")
    own = max_scale * own_mass * crystal.lattice.volume
    total = sum(products) + own
    if total <= 0:
        return DetectionLimit(crystal.name, None, strongest.two_theta,
                              strongest.hkl, "las escalas suman cero")
    return DetectionLimit(crystal.name, own / total, strongest.two_theta,
                          strongest.hkl, "")


def _typical_fwhm(phase) -> float | None:
    """The width this phase's own reflections show, in degrees."""
    for attribute in ("fwhm", "u", "width"):
        value = getattr(phase, attribute, None)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def _cell_mass(crystal) -> float | None:
    """Mass of one unit cell, in u.

    Built from `Crystal.cell_composition`, which already expands the
    symmetry and applies occupancies, rather than re-deriving the site
    multiplicities here. The first version of this module did re-derive
    them and got them wrong, which is the usual outcome of writing a
    second copy of something the package already has.
    """
    from .structure import ATOMIC_WEIGHT

    total = 0.0
    for element, number in crystal.cell_composition().items():
        weight = ATOMIC_WEIGHT.get(element)
        if weight is None:
            return None
        total += weight * number
    return total


__all__ = ["DetectionLimit", "detection_limit"]
