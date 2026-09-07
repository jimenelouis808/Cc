"""Preparing a measured diffractogram: Kα₂, background, smoothing.

Only the Kα₂ strip is subtle enough to need explaining here, and it is
worth explaining because skipping it produces a specific wrong answer
rather than a slightly worse one.

A sealed tube with only a Kβ filter emits Kα₁ and Kα₂, two lines about
0.4 % apart in wavelength with roughly 2:1 intensity. Every reflection
therefore appears twice, separated by ``Δ2θ = 2 tanθ · Δλ/λ`` — which is
0.02° at 20° 2θ and 0.20° at 80°, so the doublet is invisible at low angle
and fully resolved at high. Left in place it does three things: the peak
finder reports twice as many peaks as there are reflections, the extra
ones land in the "unexplained" list and look like an unknown phase, and a
profile fit widens the peaks to cover both — handing on a crystallite size
that is too small by a factor that grows with angle.

:func:`strip_kalpha2` removes it by Rachinger's method, which is exact
under one assumption: the two lines have identical profiles differing only
by the scale and the shift. That assumption is good for a well-aligned
laboratory diffractometer and it fails for a badly aligned one — and when
it fails it leaves a small negative dip on the high-angle side of the
strongest peaks, which is how you can see that it failed.

Rietveld refinement does not need any of this: it models both lines. Strip
only for peak finding and search-match, and refine on the raw data.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..core.baseline import asls_baseline
from .pattern import Pattern

#: Default asymmetric-least-squares stiffness for a diffraction background.
#: Stiffer than the Raman default because a diffraction background is a
#: genuinely smooth instrumental and air-scatter curve, whereas a
#: fluorescence background can bend sharply.
BACKGROUND_LAMBDA = 1e6


def kalpha2_offset(two_theta: np.ndarray, wavelength: float, wavelength2: float) -> np.ndarray:
    """Angular separation of the Kα₂ satellite, in degrees, at each 2θ.

    From ``λ = 2d sinθ`` at fixed d: ``Δθ = tanθ · Δλ/λ``.
    """
    theta = np.radians(np.asarray(two_theta, dtype=float)) / 2.0
    return np.degrees(2.0 * np.tan(theta) * (wavelength2 - wavelength) / wavelength)


def strip_kalpha2(
    pattern: Pattern, ratio: Optional[float] = None
) -> Pattern:
    """Remove the Kα₂ contribution by Rachinger's method.

    Working upwards in angle, the Kα₂ contribution at each point is the
    already-corrected Kα₁ intensity from ``Δ2θ`` below it, times the
    intensity ratio. The recursion is exact when the two lines share a
    profile.

    Returns a copy with :attr:`Pattern.kalpha2_ratio` set to zero, so
    nothing downstream strips it twice.
    """
    if not pattern.has_doublet and ratio is None:
        return pattern.copy()
    weight = pattern.kalpha2_ratio if ratio is None else float(ratio)
    if weight <= 0.0:
        return pattern.copy()

    angles = pattern.two_theta
    offsets = kalpha2_offset(angles, pattern.wavelength, pattern.kalpha2_lambda)
    corrected = np.array(pattern.intensity, dtype=float)
    for index in range(len(angles)):
        source = angles[index] - offsets[index]
        if source < angles[0]:
            continue
        # np.interp on the partially corrected array: every point below
        # this one is already free of Kalpha2, which is what makes the
        # recursion valid.
        contribution = np.interp(source, angles[:index + 1], corrected[:index + 1])
        corrected[index] -= weight * contribution

    clone = pattern.copy()
    clone.intensity = corrected
    clone.kalpha2_ratio = 0.0
    clone.name = pattern.name
    clone.history.append(f"strip_kalpha2(ratio={weight:g})")
    negative = float(np.min(corrected))
    if negative < -3.0 * pattern.noise_estimate():
        clone.metadata["kalpha2_warning"] = (
            f"la corrección de Kα₂ deja mínimos de {negative:.0f} cuentas, más "
            "negativos que el ruido. Eso significa que los dos perfiles no son "
            "iguales — desalineación del difractómetro o una rendija mal "
            "elegida — y que el pelado ha quitado de más en el flanco de alto "
            "ángulo de los picos fuertes. Para refinar, usa el patrón SIN "
            "pelar: Rietveld modela las dos líneas y no necesita esto"
        )
    return clone


def subtract_background(
    pattern: Pattern, lam: float = BACKGROUND_LAMBDA, p: float = 0.001
) -> tuple[Pattern, np.ndarray]:
    """Remove a smooth background with asymmetric least squares.

    Returns ``(corrected pattern, background)``. For **refinement**, do not
    do this: a Rietveld fit models the background with its own polynomial
    and subtracting first destroys the Poisson statistics the weighting
    depends on. This is for peak finding and for looking at the data.
    """
    background = asls_baseline(pattern.intensity, lam=lam, p=p)
    clone = pattern.copy()
    clone.intensity = pattern.intensity - background
    clone.counts = False
    clone.sigma = pattern.sigma.copy() if pattern.sigma is not None else None
    clone.sigma_origin = pattern.sigma_origin
    clone.history.append(f"subtract_background(lam={lam:g}, p={p:g})")
    return clone, background


def smooth(pattern: Pattern, window: int = 5) -> Pattern:
    """Moving average, for display only.

    Deliberately blunt and deliberately discouraged. Smoothing a
    diffractogram before refining it correlates neighbouring points, which
    makes χ² meaningless and every estimated uncertainty too small. If the
    pattern is too noisy to see, count for longer.
    """
    if window < 3:
        return pattern.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    padded = np.pad(pattern.intensity, window // 2, mode="edge")
    clone = pattern.copy()
    clone.intensity = np.convolve(padded, kernel, mode="valid")
    clone.counts = False
    clone.history.append(f"smooth({window})")
    clone.metadata["smoothed"] = (
        "Suavizado solo para ver. NO refines sobre un patrón suavizado: "
        "correlaciona puntos vecinos, y entonces la chi-cuadrado y todas las "
        "incertidumbres del ajuste dejan de significar nada."
    )
    return clone


def instrument_resolution(two_theta: float, u: float = 0.0, v: float = 0.0,
                          w: float = 0.0036) -> float:
    """Instrument FWHM in degrees, from Caglioti coefficients.

    The default is a plausible laboratory diffractometer, ``FWHM ≈ 0.06°``
    at low angle. It is a placeholder and should be replaced by the values
    refined from a line-broadening standard (LaB₆, or the silicon in the
    bundled library). Quoting a Scherrer size against a guessed instrument
    resolution is quoting the guess.
    """
    tan_theta = math.tan(math.radians(two_theta) / 2.0)
    return math.sqrt(max(u * tan_theta**2 + v * tan_theta + w, 1e-8))


__all__ = [
    "BACKGROUND_LAMBDA",
    "instrument_resolution",
    "kalpha2_offset",
    "smooth",
    "strip_kalpha2",
    "subtract_background",
]
