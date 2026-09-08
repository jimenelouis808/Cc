"""Separating crystallite size from microstrain, and carbon's own measures.

A diffraction peak is broadened by two things at once, and they can be
told apart because they depend differently on the angle:

============ ================================ =====================
size         ``β = Kλ / (D cos θ)``           falls as 1/cos θ
strain       ``β = 4ε tan θ``                 rises as tan θ
============ ================================ =====================

A single peak cannot separate them — Scherrer applied to one reflection
silently assigns *all* the broadening to size, and in a strained sample
that is wrong by whatever the strain contributes. Several reflections can,
and the three methods here are the standard ways of doing it:

**Williamson–Hall** plots ``β cos θ`` against ``sin θ``. The intercept is
the size term and the slope is the strain. It assumes both contributions
are Lorentzian and simply add, which is the weakest of the three
assumptions and the reason a Williamson–Hall plot so often has a
negative intercept — a "negative crystallite size", which is a signal
that the model does not fit, not a number to report.

**The size–strain plot** works with ``(dβcosθ)²`` against ``d²βcosθ``, on
the assumption that the size broadening is Lorentzian and the strain
Gaussian, which is closer to what is usually true. It also weights the
low-angle reflections more, where the peaks are better resolved.

**Halder–Wagner** uses ``(β*/d*)²`` against ``β*/d*²`` in reciprocal
space, with a Voigt assumption. It is the least sensitive of the three to
the high-angle reflections, which are the weakest and worst measured.

They disagree. That disagreement is information — it is the size and
strain model failing — and all three are reported rather than one being
picked.

Carbon gets its own measures, because the layer stacking is not a crystal
in the usual sense: ``d₀₀₂`` says how far apart the layers are (0.3354 nm
in graphite, larger and larger as the stacking gets more random),
``L_c`` how many layers are stacked and ``L_a`` how wide they are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .powder import instrumental_correction, scherrer

#: Interlayer spacing of perfect graphite, in Å.
GRAPHITE_D002 = 3.354

#: Interlayer spacing of fully turbostratic carbon, in Å. Above this the
#: layers have no registration with each other at all.
TURBOSTRATIC_D002 = 3.44

#: Scherrer constants. They are not one number: the shape constant depends
#: on the crystallite habit and on which width is used, and the two carbon
#: ones are the values the carbon literature has used for sixty years.
K_SPHERE = 0.89
K_LC = 0.89
"""For the 002 reflection: stack height."""
K_LA = 1.84
"""For the 100/110 reflections: layer width. The larger constant is the
two-dimensional one, and using 0.89 there gives an L_a half of what
everyone else reports."""


@dataclass
class SizeStrain:
    """Crystallite size and microstrain from a set of reflections."""

    method: str
    size_nm: Optional[float]
    strain: Optional[float]
    """Dimensionless microstrain ε. Multiply by 100 for per cent."""
    intercept: float = 0.0
    slope: float = 0.0
    r_squared: float = 0.0
    points: int = 0
    available: bool = True
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.available:
            return f"{self.method}: no disponible ({self.reason})"
        size = f"{self.size_nm:.1f} nm" if self.size_nm else "no determinado"
        strain = f"{100 * self.strain:.3f} %" if self.strain is not None else "n/d"
        return (f"{self.method}: D = {size}, ε = {strain} "
                f"(R² = {self.r_squared:.3f}, {self.points} reflexiones)")


@dataclass
class CarbonMicrostructure:
    """The four numbers a carbon diffractogram is read for."""

    d002: Optional[float] = None
    """Interlayer spacing in Å."""
    lc_nm: Optional[float] = None
    """Stack height from the 002 width."""
    la_nm: Optional[float] = None
    """Layer width from the 100 or 110 width."""
    layers: Optional[float] = None
    """``L_c / d₀₀₂``: how many layers are stacked, on average."""
    graphitisation: Optional[float] = None
    """Maire–Mering degree of graphitisation, 0 to 1. Only meaningful
    between the graphitic and turbostratic spacings."""
    turbostratic: bool = False
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = []
        if self.d002:
            parts.append(f"d₀₀₂ = {self.d002:.4f} Å")
        if self.lc_nm:
            parts.append(f"L_c = {self.lc_nm:.1f} nm")
        if self.layers:
            parts.append(f"≈ {self.layers:.0f} capas")
        if self.la_nm:
            parts.append(f"L_a = {self.la_nm:.1f} nm")
        if self.graphitisation is not None:
            parts.append(f"g = {self.graphitisation:.2f}")
        return ", ".join(parts) if parts else "sin datos suficientes"


def _prepare(
    two_theta: Sequence[float],
    fwhm_deg: Sequence[float],
    wavelength: float,
    instrument_fwhm: float,
    lorentzian: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Angles and sample-only widths, in radians, with the refusals."""
    angles = np.asarray(two_theta, dtype=float)
    widths = np.asarray(fwhm_deg, dtype=float)
    if angles.size != widths.size:
        raise ValueError(
            f"hay {angles.size} ángulos y {widths.size} anchuras")

    warnings: list[str] = []
    corrected = []
    kept = []
    for angle, width in zip(angles, widths):
        sample = instrumental_correction(width, instrument_fwhm, lorentzian)
        if sample is None:
            warnings.append(
                f"la reflexión de {angle:.2f}° no es más ancha que el "
                f"instrumento ({instrument_fwhm:.3f}°): está limitada por la "
                "resolución y no dice nada del tamaño"
            )
            continue
        corrected.append(math.radians(sample))
        kept.append(angle)
    return (np.asarray(kept, dtype=float),
            np.asarray(corrected, dtype=float), warnings)


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least squares line, with the coefficient of determination."""
    if x.size < 2:
        return 0.0, 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    total = float(np.sum((y - np.mean(y)) ** 2))
    residual = float(np.sum((y - predicted) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else 0.0
    return float(slope), float(intercept), float(r_squared)


def williamson_hall(
    two_theta: Sequence[float],
    fwhm_deg: Sequence[float],
    wavelength: float,
    instrument_fwhm: float = 0.0,
    k: float = K_SPHERE,
    lorentzian: bool = True,
) -> SizeStrain:
    """Size and strain from ``β cos θ = Kλ/D + 4ε sin θ``.

    The straight line through ``(sin θ, β cos θ)`` has the size in its
    intercept and the strain in its slope. Needs at least three
    reflections, and prefers them spread in angle: three peaks between 20
    and 30° determine a slope no better than one does.

    A **negative intercept** is not a negative crystallite size. It means
    the additive-Lorentzian assumption does not hold for this sample, and
    it is reported as such rather than as a number.
    """
    angles, widths, warnings = _prepare(two_theta, fwhm_deg, wavelength,
                                        instrument_fwhm, lorentzian)
    if angles.size < 3:
        return SizeStrain(
            method="Williamson-Hall", size_nm=None, strain=None,
            available=False, warnings=warnings,
            reason=f"hacen falta tres reflexiones y quedan {angles.size}",
        )

    theta = np.radians(angles) / 2.0
    x = np.sin(theta)
    y = widths * np.cos(theta)
    slope, intercept, r_squared = _fit(x, y)

    size = None
    if intercept > 0:
        size = 0.1 * k * wavelength / intercept
    else:
        warnings.append(
            "el corte con el eje es negativo, o sea «tamaño de cristalito "
            "negativo»: eso significa que el modelo aditivo de Lorentzianas "
            "no vale para esta muestra, no que haya un número que informar. "
            "Mira el gráfico de tamaño-deformación y Halder-Wagner"
        )
    strain = slope / 4.0
    if strain < 0:
        warnings.append(
            "la pendiente es negativa: la deformación saldría negativa, que "
            "no existe. Suele ser anisotropía —la muestra se ensancha de "
            "forma distinta según la dirección— y entonces hay que separar "
            "las reflexiones por familias"
        )
    if float(np.max(x) - np.min(x)) < 0.15:
        warnings.append(
            "las reflexiones abarcan poco ángulo: la pendiente y el corte "
            "están correlacionados y ninguno de los dos está determinado"
        )
    if r_squared < 0.7:
        warnings.append(
            f"R² = {r_squared:.2f}: los puntos no caen en una recta, así que "
            "el modelo isótropo de tamaño y deformación no describe esta "
            "muestra"
        )
    return SizeStrain(
        method="Williamson-Hall", size_nm=size, strain=strain,
        intercept=intercept, slope=slope, r_squared=r_squared,
        points=int(angles.size), warnings=warnings,
    )


def size_strain_plot(
    two_theta: Sequence[float],
    fwhm_deg: Sequence[float],
    wavelength: float,
    instrument_fwhm: float = 0.0,
    k: float = K_SPHERE,
    lorentzian: bool = True,
) -> SizeStrain:
    """Size and strain assuming Lorentzian size and Gaussian strain.

    Plots ``(d β cos θ)²`` against ``d² β cos θ``. The slope gives the
    size and the intercept the strain, which is the other way round from
    Williamson–Hall — and it means the size, the number usually wanted,
    comes from the better-determined of the two parameters.
    """
    angles, widths, warnings = _prepare(two_theta, fwhm_deg, wavelength,
                                        instrument_fwhm, lorentzian)
    if angles.size < 3:
        return SizeStrain(
            method="Tamaño-deformación", size_nm=None, strain=None,
            available=False, warnings=warnings,
            reason=f"hacen falta tres reflexiones y quedan {angles.size}",
        )

    theta = np.radians(angles) / 2.0
    d = wavelength / (2.0 * np.sin(theta))
    beta_cos = widths * np.cos(theta)
    x = d ** 2 * beta_cos
    y = (d * beta_cos) ** 2
    slope, intercept, r_squared = _fit(x, y)

    size = 0.1 * k * wavelength / slope if slope > 0 else None
    if size is None:
        warnings.append(
            "la pendiente no es positiva: no sale un tamaño de este gráfico")
    strain = math.sqrt(intercept) / 2.0 if intercept > 0 else 0.0
    if intercept < 0:
        warnings.append(
            "el corte es negativo, o sea deformación nula dentro del error; "
            "se informa cero, no un número imaginario"
        )
    return SizeStrain(
        method="Tamaño-deformación", size_nm=size, strain=strain,
        intercept=intercept, slope=slope, r_squared=r_squared,
        points=int(angles.size), warnings=warnings,
    )


def halder_wagner(
    two_theta: Sequence[float],
    fwhm_deg: Sequence[float],
    wavelength: float,
    instrument_fwhm: float = 0.0,
    k: float = K_SPHERE,
    lorentzian: bool = True,
) -> SizeStrain:
    """Size and strain in reciprocal space, with a Voigt assumption.

    ``(β*/d*)² = (1/D)(β*/d*²) + (ε/2)²``, with ``β* = β cos θ / λ`` and
    ``d* = 2 sin θ / λ``. Weights the low-angle reflections most, which is
    where the peaks are strongest and best resolved — the opposite of
    Williamson–Hall, which lets a weak, badly-measured high-angle peak set
    the slope.
    """
    angles, widths, warnings = _prepare(two_theta, fwhm_deg, wavelength,
                                        instrument_fwhm, lorentzian)
    if angles.size < 3:
        return SizeStrain(
            method="Halder-Wagner", size_nm=None, strain=None,
            available=False, warnings=warnings,
            reason=f"hacen falta tres reflexiones y quedan {angles.size}",
        )

    theta = np.radians(angles) / 2.0
    beta_star = widths * np.cos(theta) / wavelength
    d_star = 2.0 * np.sin(theta) / wavelength
    x = beta_star / d_star ** 2
    y = (beta_star / d_star) ** 2
    slope, intercept, r_squared = _fit(x, y)

    size = 0.1 / slope if slope > 0 else None
    if size is None:
        warnings.append("la pendiente no es positiva: no sale un tamaño")
    strain = 2.0 * math.sqrt(intercept) if intercept > 0 else 0.0
    return SizeStrain(
        method="Halder-Wagner", size_nm=size, strain=strain,
        intercept=intercept, slope=slope, r_squared=r_squared,
        points=int(angles.size), warnings=warnings,
    )


def compare_methods(
    two_theta: Sequence[float],
    fwhm_deg: Sequence[float],
    wavelength: float,
    instrument_fwhm: float = 0.0,
    k: float = K_SPHERE,
    lorentzian: bool = True,
) -> list[SizeStrain]:
    """All three, because they disagree and the disagreement is the point.

    Three sizes within a few per cent of each other mean the size-strain
    model fits. Sizes differing by a factor of two mean it does not, and
    then no single number should be quoted — which is a result, and a more
    useful one than a confident wrong figure.
    """
    arguments = (two_theta, fwhm_deg, wavelength, instrument_fwhm, k, lorentzian)
    return [williamson_hall(*arguments), size_strain_plot(*arguments),
            halder_wagner(*arguments)]


def agreement(results: Sequence[SizeStrain]) -> Optional[str]:
    """What the three methods' disagreement says, in one sentence."""
    sizes = [r.size_nm for r in results if r.available and r.size_nm]
    if len(sizes) < 2:
        return ("solo un método ha dado un tamaño: eso ya dice que el modelo "
                "de tamaño y deformación no describe bien esta muestra")
    spread = (max(sizes) - min(sizes)) / max(np.mean(sizes), 1e-9)
    if spread < 0.15:
        return (f"los métodos coinciden dentro del {100 * spread:.0f} %: el "
                "modelo de tamaño y deformación describe la muestra")
    if spread < 0.5:
        return (f"los métodos difieren un {100 * spread:.0f} %: informa el "
                "intervalo, no un número")
    return (f"los métodos difieren un {100 * spread:.0f} %, más de la mitad: "
            "el ensanchamiento no es el de un tamaño con deformación "
            "isótropa. Suele ser anisotropía de forma o una distribución "
            "ancha de tamaños, y entonces ningún número único vale")


# -- carbon -------------------------------------------------------------

def carbon_microstructure(
    two_theta_002: Optional[float] = None,
    fwhm_002: Optional[float] = None,
    two_theta_100: Optional[float] = None,
    fwhm_100: Optional[float] = None,
    wavelength: float = 1.540598,
    instrument_fwhm: float = 0.0,
) -> CarbonMicrostructure:
    """d₀₀₂, L_c, L_a and the degree of graphitisation.

    The measures the carbon literature uses, with the constants it uses:
    K = 0.89 for the stack height from 002 and **K = 1.84** for the layer
    width from 100. That second constant is the two-dimensional one and
    using 0.89 there — the obvious thing — gives an L_a half of everyone
    else's.

    The degree of graphitisation is Maire–Mering,
    ``g = (3.440 − d₀₀₂)/(3.440 − 3.354)``, and it is only meaningful
    between those two spacings. Outside them it is not a percentage of
    anything, so it is refused rather than clipped.
    """
    result = CarbonMicrostructure()
    theta = None

    if two_theta_002:
        theta = math.radians(two_theta_002) / 2.0
        result.d002 = wavelength / (2.0 * math.sin(theta))
        if fwhm_002:
            sample = instrumental_correction(fwhm_002, instrument_fwhm)
            if sample is None:
                result.warnings.append(
                    "la 002 no es más ancha que el instrumento: L_c estaría "
                    "midiendo la resolución del equipo"
                )
            else:
                result.lc_nm = scherrer(sample, two_theta_002, wavelength,
                                        k=K_LC)
                result.layers = result.lc_nm * 10.0 / result.d002

    if two_theta_100 and fwhm_100:
        sample = instrumental_correction(fwhm_100, instrument_fwhm)
        if sample is None:
            result.warnings.append(
                "la 100 no es más ancha que el instrumento: no sale L_a")
        else:
            result.la_nm = scherrer(sample, two_theta_100, wavelength, k=K_LA)

    if result.d002:
        spacing = result.d002
        result.turbostratic = spacing >= TURBOSTRATIC_D002
        if GRAPHITE_D002 <= spacing <= TURBOSTRATIC_D002:
            result.graphitisation = ((TURBOSTRATIC_D002 - spacing)
                                     / (TURBOSTRATIC_D002 - GRAPHITE_D002))
        elif spacing > TURBOSTRATIC_D002:
            result.warnings.append(
                f"d₀₀₂ = {spacing:.4f} Å está por encima del valor "
                f"turbostrático ({TURBOSTRATIC_D002} Å): el apilamiento no "
                "tiene ningún registro y el grado de grafitización no mide "
                "nada ahí"
            )
        else:
            result.warnings.append(
                f"d₀₀₂ = {spacing:.4f} Å es MENOR que el del grafito "
                f"({GRAPHITE_D002} Å): eso no pasa en un carbono, así que hay "
                "un error de cero o de desplazamiento de muestra, y ese error "
                "va entero a L_c también"
            )
        if result.turbostratic:
            result.warnings.append(
                "con apilamiento turbostrático, la 002 no es una reflexión de "
                "un cristal tridimensional y L_c es una longitud de "
                "coherencia, no un tamaño de cristalito"
            )
    if result.lc_nm and result.lc_nm > 120.0:
        result.warnings.append(
            "L_c por encima de 120 nm: ahí el ensanchamiento por tamaño ya es "
            "menor que la resolución y el número lo fija lo que hayas supuesto "
            "para el instrumento"
        )
    return result


def amorphous_fraction(
    crystalline_weights: dict[str, float],
    standard: str,
    standard_added: float,
) -> dict[str, float]:
    """Amorphous content from an internal standard.

    Rietveld weight fractions are of the crystalline, *modelled* part and
    always sum to 100 % — an amorphous halo contributes nothing to them
    and its intensity is shared out among the phases that were modelled.
    The only way to measure what is missing is to add a known amount of a
    known crystalline standard and see how much the refinement thinks
    there is: the excess is the amorphous fraction.

    Parameters
    ----------
    crystalline_weights:
        Refined weight fractions, as fractions summing to 1.
    standard:
        Which of them is the standard.
    standard_added:
        The weight fraction of standard actually added, 0 to 1.

    Returns
    -------
    dict
        ``amorfo``, and every phase's weight fraction of the whole sample.
    """
    if standard not in crystalline_weights:
        raise ValueError(
            f"«{standard}» no está entre las fases: "
            + ", ".join(sorted(crystalline_weights))
        )
    if not 0 < standard_added < 1:
        raise ValueError("la fracción de patrón añadida va entre 0 y 1")
    refined = crystalline_weights[standard]
    if refined <= 0:
        raise ValueError("el patrón ha refinado a cero: revisa el ajuste")

    # The refinement over-states the standard by exactly the factor by
    # which everything crystalline is over-stated.
    scale = standard_added / refined
    amorphous = 1.0 - scale
    result = {
        name: weight * scale
        for name, weight in crystalline_weights.items()
        if name != standard
    }
    total = sum(result.values())
    if total > 0:
        # Re-express on a standard-free basis, which is what is wanted:
        # the standard is not part of the sample.
        divisor = 1.0 - standard_added
        result = {name: value / divisor for name, value in result.items()}
        amorphous = amorphous / divisor
    result["amorfo"] = max(0.0, amorphous)
    return result


__all__ = [
    "GRAPHITE_D002",
    "K_LA",
    "K_LC",
    "K_SPHERE",
    "TURBOSTRATIC_D002",
    "CarbonMicrostructure",
    "SizeStrain",
    "agreement",
    "amorphous_fraction",
    "carbon_microstructure",
    "compare_methods",
    "halder_wagner",
    "size_strain_plot",
    "williamson_hall",
]
