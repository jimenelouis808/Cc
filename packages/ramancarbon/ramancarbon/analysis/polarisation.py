"""Polarised Raman: mode symmetry, and how aligned the sample is.

Two different measurements share one geometry.

**Depolarisation ratio.** With the incident light polarised one way and
the analyser parallel or crossed, the ratio ρ = I⊥/I∥ separates totally
symmetric modes (ρ < 0.75) from every other mode (ρ = 0.75 exactly, for
randomly oriented molecules and non-resonant scattering). It is how a
band is assigned to a symmetry species without a calculation.

**Angular dependence.** Rotating the sample, or the polarisation, through
a series of angles measures alignment: aligned nanotubes scatter as
cos⁴ of the angle between their axis and the polarisation, and the
modulation depth is the order parameter.

The trap in both is the same and it is instrumental: **a grating is
itself a polariser.** Its diffraction efficiency differs between s and p
by tens of per cent, so a measured ρ is the sample's ratio times the
spectrometer's, and the spectrometer's is usually the larger effect.
A scrambler in front of the slit removes it; a measured correction factor
also removes it. Nothing here computes ρ without saying which of those
was done.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..core.compat import trapezoid
from ..core.spectrum import Spectrum

#: The classical limit for non-resonant scattering from randomly oriented
#: scatterers. A band above it is resonant, oriented, or leaking.
DEPOLARISATION_LIMIT = 0.75


@dataclass
class Depolarisation:
    """One band's depolarisation ratio and what it implies."""

    ratio: Optional[float]
    band_cm: float = 0.0
    parallel_area: float = 0.0
    perpendicular_area: float = 0.0
    symmetry: str = ""
    """``"totalmente simétrico"``, ``"despolarizado"`` or ``"anómalo"``."""
    corrected: bool = False
    available: bool = True
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.available or self.ratio is None:
            return f"ρ no disponible: {self.reason}"
        text = f"ρ = {self.ratio:.3f} en {self.band_cm:.0f} cm⁻¹ → {self.symmetry}"
        if not self.corrected:
            text += " (sin corregir la respuesta polarizada del equipo)"
        return text


@dataclass
class Alignment:
    """How aligned an anisotropic sample is, from an angular series."""

    angle_deg: Optional[float]
    """Where the intensity is greatest: the alignment direction."""
    modulation: Optional[float] = None
    """``(I_max − I_min) / (I_max + I_min)``, between 0 and 1."""
    order_parameter: Optional[float] = None
    """⟨P₂⟩ from the modulation, under the cos⁴ model. 0 is random, 1 is
    perfectly aligned."""
    model: str = "cos4"
    residual: float = 0.0
    available: bool = True
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.available or self.angle_deg is None:
            return f"alineación no disponible: {self.reason}"
        text = (f"eje a {self.angle_deg:.1f}°, modulación "
                f"{self.modulation:.2f}")
        if self.order_parameter is not None:
            text += f", ⟨P₂⟩ ≈ {self.order_parameter:.2f}"
        return text


def _area(spectrum: Spectrum, low: float, high: float) -> float:
    """Band area with a straight local background removed."""
    x, y = spectrum.region(low, high)
    if x.size < 5:
        return 0.0
    edge = max(2, x.size // 10)
    left, right = float(np.mean(y[:edge])), float(np.mean(y[-edge:]))
    fraction = (x - x[0]) / max(x[-1] - x[0], 1e-12)
    return float(trapezoid(y - (left + (right - left) * fraction), x))


def depolarisation_ratio(
    parallel: Spectrum,
    perpendicular: Spectrum,
    band_cm: float,
    window: float = 40.0,
    correction: Optional[float] = None,
    scrambler: bool = False,
) -> Depolarisation:
    """ρ = I⊥/I∥ for one band.

    Parameters
    ----------
    parallel, perpendicular:
        The two geometries, acquired with the **same** exposure and
        accumulations. They are not renormalised here: scaling one of them
        to match the other is scaling away the measurement.
    correction:
        The instrument's own ⊥/∥ throughput ratio, measured on a source of
        known polarisation. Divides it out.
    scrambler:
        Say ``True`` when a polarisation scrambler was in the beam path,
        in which case no correction is needed.

    Returns
    -------
    Depolarisation
        ``available=False`` with a reason when the band is not there.
    """
    if parallel.laser_nm and perpendicular.laser_nm and \
            parallel.laser_nm != perpendicular.laser_nm:
        return Depolarisation(
            ratio=None, band_cm=band_cm, available=False,
            reason="los dos espectros están tomados con láseres distintos",
        )

    low, high = band_cm - window, band_cm + window
    for name, spectrum in (("paralelo", parallel), ("perpendicular", perpendicular)):
        if not spectrum.covers(low, high):
            return Depolarisation(
                ratio=None, band_cm=band_cm, available=False,
                reason=f"el espectro {name} no cubre {low:.0f}–{high:.0f} cm⁻¹",
            )

    par = _area(parallel, low, high)
    perp = _area(perpendicular, low, high)
    if par <= 0:
        return Depolarisation(
            ratio=None, band_cm=band_cm, parallel_area=par,
            perpendicular_area=perp, available=False,
            reason="la banda no tiene área positiva en la geometría paralela",
        )

    ratio = perp / par
    if correction:
        ratio /= correction

    warnings: list[str] = []
    if ratio < 0.72:
        symmetry = "totalmente simétrico"
    elif ratio <= 0.78:
        symmetry = "despolarizado"
    else:
        symmetry = "anómalo"
        warnings.append(
            f"ρ = {ratio:.2f} supera el límite clásico de "
            f"{DEPOLARISATION_LIMIT}: en dispersión no resonante de "
            "orientaciones al azar eso no puede pasar. Lo habitual es fuga "
            "del analizador, respuesta polarizada de la red, o dispersión "
            "resonante — que sí lo permite"
        )
    if not (correction or scrambler):
        warnings.append(
            "sin mezclador de polarización ni factor de corrección: una red "
            "de difracción es ella misma un polarizador y su eficiencia "
            "difiere entre s y p en decenas por ciento, así que este ρ lleva "
            "dentro el del espectrómetro"
        )

    noise = max(parallel.noise_estimate(), perpendicular.noise_estimate())
    if par < 10 * noise * math.sqrt(2 * window):
        warnings.append(
            "la banda está cerca del ruido en la geometría paralela; ρ es un "
            "cociente y su error crece sin límite cuando el denominador baja"
        )
    return Depolarisation(
        ratio=ratio, band_cm=band_cm, parallel_area=par, perpendicular_area=perp,
        symmetry=symmetry, corrected=bool(correction or scrambler),
        warnings=warnings,
    )


def angular_alignment(
    angles_deg: Sequence[float],
    spectra: Sequence[Spectrum],
    band_cm: float,
    window: float = 40.0,
    model: str = "cos4",
) -> Alignment:
    """Alignment from a series of spectra taken at different angles.

    The intensity of an aligned one-dimensional scatterer follows cos⁴ of
    the angle between its axis and the polarisation, in the parallel
    geometry — cos² for the incident field and cos² again for the
    scattered one. Using cos² instead, which is the common shortcut, gives
    an order parameter about twice too large.

    Fitted by least squares on the two linear coefficients after fixing
    the phase from the discrete maximum, which needs no starting guess and
    cannot fall into a wrong minimum.
    """
    angles = np.asarray(angles_deg, dtype=float)
    if angles.size != len(spectra):
        return Alignment(angle_deg=None, available=False,
                         reason=f"hay {angles.size} ángulos y {len(spectra)} "
                                "espectros")
    if angles.size < 4:
        return Alignment(
            angle_deg=None, available=False,
            reason="hacen falta al menos cuatro ángulos: con tres, cualquier "
                   "modulación se ajusta perfectamente y no significa nada",
        )

    areas = np.asarray([_area(s, band_cm - window, band_cm + window)
                        for s in spectra], dtype=float)
    if not np.all(np.isfinite(areas)) or np.max(areas) <= 0:
        return Alignment(angle_deg=None, available=False,
                         reason="la banda no aparece en la serie angular")

    power = 4 if model == "cos4" else 2
    # Scan the phase on a fine grid and take the least-squares fit of
    # a + b·cosᵖ(θ−θ₀) at each: the model is linear in a and b once θ₀ is
    # fixed, so there is no iteration and no local minimum.
    best = None
    for phase in np.arange(0.0, 180.0, 0.5):
        basis = np.cos(np.radians(angles - phase)) ** power
        design = np.stack([np.ones_like(basis), basis], axis=1)
        solution, *_ = np.linalg.lstsq(design, areas, rcond=None)
        residual = float(np.sum((design @ solution - areas) ** 2))
        if solution[1] < 0:
            continue
        if best is None or residual < best[0]:
            best = (residual, phase, solution)
    if best is None:
        return Alignment(angle_deg=None, available=False,
                         reason="la serie no tiene modulación positiva: la "
                                "muestra no está alineada o el ajuste no "
                                "encuentra un eje")

    residual, phase, (offset, amplitude) = best
    maximum, minimum = offset + amplitude, offset
    modulation = ((maximum - minimum) / (maximum + minimum)
                  if maximum + minimum > 0 else 0.0)
    total = float(np.sum((areas - np.mean(areas)) ** 2))
    relative = math.sqrt(residual / total) if total > 0 else 0.0

    # ⟨P₂⟩ from the modulation under the cos⁴ model. Approximate, and
    # labelled as such: the exact inversion needs the full orientation
    # distribution, which one band at one geometry does not determine.
    order = min(1.0, max(0.0, modulation / (1.0 + 0.5 * modulation)))

    warnings = [
        "⟨P₂⟩ sale de la modulación bajo el modelo cos⁴ y es aproximado: la "
        "inversión exacta necesita la distribución de orientaciones entera, "
        "que una banda en una sola geometría no determina"
    ]
    if angles.max() - angles.min() < 90.0:
        warnings.append(
            f"la serie solo abarca {angles.max() - angles.min():.0f}°; para "
            "fijar el eje hacen falta al menos 90°, y 180° es lo seguro"
        )
    if relative > 0.3:
        warnings.append(
            f"el ajuste deja un residuo del {100 * relative:.0f} %: la "
            f"dependencia angular no es cos^{power}, así que el eje que sale "
            "es poco firme"
        )
    return Alignment(
        angle_deg=float(phase % 180.0), modulation=float(modulation),
        order_parameter=float(order), model=model, residual=relative,
        warnings=warnings,
    )


__all__ = [
    "DEPOLARISATION_LIMIT",
    "Alignment",
    "Depolarisation",
    "angular_alignment",
    "depolarisation_ratio",
]
