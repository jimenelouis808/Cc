"""The measured diffractogram.

Deliberately the same shape as :class:`ramancarbon.core.spectrum.Spectrum`:
a sorted axis, an intensity, validation on construction, and a noise
estimate that does not need a model. What differs is what the axis means,
and one consequence of that is worth stating: counting statistics.

A diffractogram is usually **counts**, and counts have Poisson noise, so
the uncertainty on a point is √N and varies by an order of magnitude across
the pattern. Rietveld refinement weights by 1/σ² for exactly that reason,
and getting it wrong biases the fit towards the tallest peak. So this class
carries an explicit :attr:`sigma`: measured e.s.d.s when the file had them
(``.xye``), √N when the data are counts, and a flat estimate otherwise —
and it records which of the three it is, because a refinement's R-factors
mean different things in each case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


class PatternError(ValueError):
    """Raised when a diffractogram is malformed."""


@dataclass
class Pattern:
    """One powder diffraction pattern."""

    two_theta: np.ndarray
    intensity: np.ndarray
    wavelength: float = 1.540598
    """In Å, the **Kα₁** line. Copper by default.

    Not the Kα₁₂ mean. A tube emits two lines and the pattern contains
    both; modelling that explicitly through :attr:`kalpha2_ratio` is
    correct at every angle, whereas a single averaged wavelength is a
    compromise that is right nowhere — it splits the difference at low
    angle and leaves a visibly asymmetric peak at high angle, which a
    profile fit then absorbs into the width and hands on as a wrong
    crystallite size."""
    sigma: Optional[np.ndarray] = None
    name: str = "patrón"
    anode: str = "Cu"
    counts: bool = True
    """Whether the intensities are raw counts, which decides whether √N is
    a legitimate uncertainty."""
    kalpha2_ratio: float = 0.5
    """Intensity of Kα₂ relative to Kα₁. Set to 0 for monochromated or
    synchrotron data — leaving it at 0.5 there invents a satellite that is
    not in the data and biases every fitted width."""
    kalpha2_wavelength: Optional[float] = None
    """Kα₂ wavelength in Å. Derived from :attr:`anode` when ``None``."""
    sigma_origin: str = "poisson"
    """``"measured"``, ``"poisson"`` or ``"flat"``."""
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        angles = np.asarray(self.two_theta, dtype=float)
        values = np.asarray(self.intensity, dtype=float)
        if angles.ndim != 1 or values.ndim != 1:
            raise PatternError("2θ e intensidad tienen que ser vectores")
        if angles.size != values.size:
            raise PatternError(
                f"2θ tiene {angles.size} puntos e intensidad {values.size}"
            )
        if angles.size < 5:
            raise PatternError(
                f"un difractograma de {angles.size} puntos no da para nada"
            )
        for label, array in (("2θ", angles), ("intensidad", values)):
            bad = np.flatnonzero(~np.isfinite(array))
            if bad.size:
                raise PatternError(
                    f"{label} contiene valores no finitos; el primero está en "
                    f"el punto {int(bad[0])}"
                )
        if np.any(angles < 0.0) or np.any(angles > 180.0):
            raise PatternError("hay valores de 2θ fuera del rango físico 0–180°")

        order = np.argsort(angles, kind="stable")
        angles, values = angles[order], values[order]
        if self.sigma is not None:
            errors = np.asarray(self.sigma, dtype=float)[order]
            if errors.size != angles.size:
                raise PatternError("sigma no tiene la misma longitud que el patrón")
            self.sigma = np.maximum(errors, 1e-9)
            self.sigma_origin = "measured"
        self.two_theta = angles
        self.intensity = values
        if self.wavelength <= 0.0:
            raise PatternError("la longitud de onda tiene que ser positiva")
        if self.sigma is None:
            self.sigma = self._default_sigma()

    def _default_sigma(self) -> np.ndarray:
        """√N for counts, a flat estimate otherwise."""
        if self.counts:
            self.sigma_origin = "poisson"
            return np.sqrt(np.maximum(self.intensity, 1.0))
        self.sigma_origin = "flat"
        return np.full(self.intensity.shape, max(self.noise_estimate(), 1e-9))

    # -- geometry -----------------------------------------------------

    @property
    def range(self) -> tuple[float, float]:
        return float(self.two_theta[0]), float(self.two_theta[-1])

    @property
    def step(self) -> float:
        """Median step in degrees."""
        return float(np.median(np.diff(self.two_theta))) if self.two_theta.size > 1 else 0.0

    @property
    def n(self) -> int:
        return int(self.two_theta.size)

    @property
    def kalpha2_lambda(self) -> float:
        """The Kα₂ wavelength actually in use, in Å."""
        if self.kalpha2_wavelength:
            return float(self.kalpha2_wavelength)
        from .scattering import ANODES

        entry = ANODES.get(self.anode)
        if entry:
            return float(entry["ka2"])
        return self.wavelength * 1.002486

    @property
    def has_doublet(self) -> bool:
        return self.kalpha2_ratio > 1e-6

    def d_spacing(self) -> np.ndarray:
        """d in Å for every point, ``inf`` at 2θ = 0."""
        sine = np.sin(np.radians(self.two_theta) / 2.0)
        with np.errstate(divide="ignore"):
            spacing = self.wavelength / (2.0 * sine)
        spacing[~np.isfinite(spacing)] = np.inf
        return spacing

    def q(self) -> np.ndarray:
        """Scattering vector magnitude ``4π sinθ/λ`` in Å⁻¹."""
        return 4.0 * math.pi * np.sin(np.radians(self.two_theta) / 2.0) / self.wavelength

    def covers(self, low: float, high: float, fraction: float = 0.9) -> bool:
        """Whether the pattern really measured a 2θ window.

        Same purpose as its Raman counterpart: distinguishing "the phase is
        absent" from "its reflections were never scanned". A phase called
        absent on a 20–60° scan when its only strong line is at 65° is a
        wrong answer, not a missing one.
        """
        low, high = float(low), float(high)
        if high <= low:
            return False
        start, stop = self.range
        overlap = max(0.0, min(high, stop) - max(low, start))
        return overlap >= fraction * (high - low)

    def noise_estimate(self) -> float:
        """Robust σ from the median absolute second difference.

        The second difference kills any smooth background and any linear
        ramp, so what is left is noise plus curvature at the peaks; the
        median is insensitive to the latter. The 1/√6 converts the second
        difference of independent noise back to the noise itself.
        """
        if self.intensity.size < 3:
            return 0.0
        second = np.diff(self.intensity, n=2)
        mad = float(np.median(np.abs(second - np.median(second))))
        return 1.4826 * mad / math.sqrt(6.0)

    def crop(self, low: float, high: float) -> "Pattern":
        """A copy restricted to a 2θ window."""
        mask = (self.two_theta >= low) & (self.two_theta <= high)
        if mask.sum() < 5:
            raise PatternError(
                f"recortar a {low}–{high}° deja {int(mask.sum())} puntos"
            )
        clone = self.copy()
        clone.two_theta = self.two_theta[mask]
        clone.intensity = self.intensity[mask]
        clone.sigma = self.sigma[mask] if self.sigma is not None else None
        clone.history.append(f"crop({low:g}, {high:g})")
        return clone

    def copy(self) -> "Pattern":
        return Pattern(
            two_theta=self.two_theta.copy(),
            intensity=self.intensity.copy(),
            wavelength=self.wavelength,
            sigma=None if self.sigma is None else self.sigma.copy(),
            name=self.name,
            anode=self.anode,
            counts=self.counts,
            kalpha2_ratio=self.kalpha2_ratio,
            kalpha2_wavelength=self.kalpha2_wavelength,
            metadata=dict(self.metadata),
            history=list(self.history),
        )

    def describe(self) -> str:
        origin = {
            "measured": "e.s.d. del archivo",
            "poisson": "√N (cuentas)",
            "flat": "estimación plana (los datos no son cuentas)",
        }[self.sigma_origin]
        return (
            f"{self.name}: {self.n} puntos, {self.range[0]:.3f}–{self.range[1]:.3f}° "
            f"2θ, paso {self.step:.4f}°, λ={self.wavelength:.6f} Å "
            f"({self.anode} Kα₁); σ = {origin}"
            + (
                f", Kα₂ presente (r={self.kalpha2_ratio:g})"
                if self.has_doublet
                else ", sin Kα₂"
            )
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Pattern(name={self.name!r}, n={self.n}, "
            f"range={self.range}, wavelength={self.wavelength})"
        )


__all__ = ["Pattern", "PatternError"]
