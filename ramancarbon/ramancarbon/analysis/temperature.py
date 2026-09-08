"""Sample temperature from the anti-Stokes to Stokes ratio.

Every Raman measurement heats the sample, and the amount is the single
most under-reported source of error in the literature: a band that has
shifted by 5 cm⁻¹ because the laser cooked the spot is routinely reported
as strain or doping. The spectrum itself says what the temperature was —
the anti-Stokes side exists only because the mode was already excited, and
how much of it there is measures the population.

.. math::

    \\frac{I_{AS}}{I_S} = \\left(\\frac{\\nu_L + \\nu_m}
    {\\nu_L - \\nu_m}\\right)^{n} \\exp\\!\\left(-\\frac{hc\\nu_m}{k_B T}\\right)

Two things about that expression decide whether the number that comes out
means anything, and both are the caller's to get right:

**The exponent is not settled.** It is 4 for scattered *power* and 3 for a
photon-counting detector, which is what a CCD is. The difference is a few
per cent in the prefactor and tens of kelvin in the answer. The default
here is 3, for a CCD; the parameter exists because the convention has to
be stated, not assumed.

**The instrument response is not symmetric.** The grating, the filters and
the detector all have a different throughput at the anti-Stokes
wavelength than at the Stokes one, and an edge filter cuts the
anti-Stokes side hard. Without a response calibration measured on a
lamp, the ratio is the instrument's as much as the sample's. This module
refuses to hide that: with no ``response`` given, the result carries the
warning and is labelled apparent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


from ..core.spectrum import Spectrum

#: hc/k_B in cm·K. One wavenumber is 1.4388 K of thermal energy.
HC_OVER_K = 1.4387768775

#: Below this the anti-Stokes band of a hard mode is unmeasurable.
MIN_RATIO = 1e-4


@dataclass
class Temperature:
    """A temperature from one mode, with everything that qualifies it."""

    kelvin: Optional[float]
    celsius: Optional[float] = None
    mode_cm: float = 0.0
    ratio: Optional[float] = None
    """The measured anti-Stokes / Stokes intensity ratio."""
    uncertainty_k: Optional[float] = None
    available: bool = True
    reason: str = ""
    exponent: int = 3
    calibrated: bool = False
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kelvin is not None and self.celsius is None:
            self.celsius = self.kelvin - 273.15

    def describe(self) -> str:
        if not self.available or self.kelvin is None:
            return f"temperatura no disponible: {self.reason}"
        text = f"T = {self.kelvin:.0f} K ({self.celsius:.0f} °C)"
        if self.uncertainty_k:
            text += f" ± {self.uncertainty_k:.0f} K"
        text += f", del modo de {self.mode_cm:.0f} cm⁻¹"
        if not self.calibrated:
            text += " (aparente: sin calibración de respuesta)"
        return text


def _band_area(spectrum: Spectrum, centre: float, window: float) -> float:
    """A band's area with its own local background removed.

    Not optional. A flat offset of 20 counts under a Stokes band of 1000
    and an anti-Stokes band of 400 raises the ratio by 8 %, and 8 % on the
    ratio is 25 K at room temperature — in the direction that makes every
    sample look hotter than it is. The background is a straight line
    through the window's own ends, where the band is not.
    """
    import numpy as np

    x, y = spectrum.region(centre - window, centre + window)
    if x.size < 5:
        return 0.0
    edge = max(2, x.size // 10)
    left = float(np.mean(y[:edge]))
    right = float(np.mean(y[-edge:]))
    fraction = (x - x[0]) / max(x[-1] - x[0], 1e-12)
    baseline = left + (right - left) * fraction

    from ..core.compat import trapezoid

    return float(trapezoid(y - baseline, x))


def stokes_anti_stokes_temperature(
    spectrum: Spectrum,
    mode_cm: float,
    window: float = 30.0,
    exponent: int = 3,
    response: Optional[float] = None,
    laser_nm: Optional[float] = None,
) -> Temperature:
    """Temperature from one mode's anti-Stokes / Stokes ratio.

    Parameters
    ----------
    spectrum:
        Must cover **both** signs of the shift. A spectrum that starts at
        +100 cm⁻¹ cannot give a temperature and is refused rather than
        extrapolated.
    mode_cm:
        The mode to use, as a positive shift. Low-frequency modes are the
        usable ones near room temperature: at 300 K a 1580 cm⁻¹ band has
        an anti-Stokes side 10⁻³ of its Stokes one, and a 200 cm⁻¹ band
        has 0.4.
    window:
        Half-width in cm⁻¹ over which the two bands are integrated.
    exponent:
        3 for a photon-counting detector, 4 for scattered power.
    response:
        The instrument's anti-Stokes/Stokes throughput ratio at this mode,
        from a calibration lamp. Without it the result is apparent and
        says so.
    laser_nm:
        Overrides the spectrum's own excitation wavelength.

    Returns
    -------
    Temperature
        With ``available=False`` and a reason when it cannot be measured.
    """
    laser = laser_nm or spectrum.laser_nm
    if not laser:
        return Temperature(
            kelvin=None, mode_cm=mode_cm, available=False,
            reason="hace falta la longitud de onda del láser para el factor "
                   "de frecuencia",
        )
    mode = abs(float(mode_cm))
    if mode <= 0:
        return Temperature(kelvin=None, mode_cm=mode, available=False,
                           reason="el modo tiene que ser un desplazamiento no nulo")

    low, high = float(spectrum.shift[0]), float(spectrum.shift[-1])
    if low > -(mode + window) or high < mode + window:
        return Temperature(
            kelvin=None, mode_cm=mode, available=False,
            reason=(f"el espectro va de {low:.0f} a {high:.0f} cm⁻¹ y hacen "
                    f"falta las dos ramas, de −{mode + window:.0f} a "
                    f"+{mode + window:.0f}"),
        )

    stokes = _band_area(spectrum, mode, window)
    anti = _band_area(spectrum, -mode, window)
    if stokes <= 0:
        return Temperature(kelvin=None, mode_cm=mode, available=False,
                           reason="la banda Stokes no tiene área positiva")
    ratio = anti / stokes
    warnings: list[str] = []

    if ratio <= MIN_RATIO:
        return Temperature(
            kelvin=None, mode_cm=mode, ratio=ratio, available=False,
            reason=(f"el cociente anti-Stokes/Stokes es {ratio:.2e}: la rama "
                    "anti-Stokes de este modo está por debajo del ruido, y a "
                    "esa altura el resultado sería el del fondo"),
        )

    corrected = ratio / response if response else ratio
    laser_cm = 1e7 / float(laser)
    prefactor = ((laser_cm + mode) / (laser_cm - mode)) ** exponent

    argument = prefactor / corrected
    if argument <= 1.0:
        return Temperature(
            kelvin=None, mode_cm=mode, ratio=ratio, available=False,
            exponent=exponent, calibrated=response is not None,
            reason=("el cociente medido supera el límite clásico: o la "
                    "respuesta del equipo no está corregida, o hay señal "
                    "ajena en una de las dos ventanas"),
        )
    kelvin = HC_OVER_K * mode / math.log(argument)

    if response is None:
        warnings.append(
            "sin calibración de la respuesta del equipo, el cociente lleva "
            "dentro la transmisión relativa de la red, los filtros y el "
            "detector entre las dos ramas; con filtro de borde el sesgo es de "
            "cientos de kelvin y siempre hacia arriba"
        )
    if mode > 800.0:
        warnings.append(
            f"un modo de {mode:.0f} cm⁻¹ tiene la rama anti-Stokes muy débil "
            "a temperatura ambiente; los modos de baja frecuencia dan una "
            "medida mucho más firme"
        )
    if kelvin > 1200.0:
        warnings.append(
            "una temperatura por encima de 1200 K en un experimento Raman "
            "corriente casi siempre significa fondo dentro de la ventana "
            "anti-Stokes, no una muestra al rojo"
        )

    noise = spectrum.noise_estimate()
    span = 2 * window
    sigma_ratio = (noise * math.sqrt(span) * math.sqrt(2.0) / stokes
                   if stokes > 0 else None)
    uncertainty = None
    if sigma_ratio:
        # dT/T = (T k / hc ν) · dR/R, from differentiating the exponential.
        relative = sigma_ratio / max(corrected, 1e-30)
        uncertainty = kelvin * relative * kelvin / (HC_OVER_K * mode)

    return Temperature(
        kelvin=kelvin, mode_cm=mode, ratio=ratio, exponent=exponent,
        calibrated=response is not None, uncertainty_k=uncertainty,
        warnings=warnings,
    )


def laser_heating(
    cold: Spectrum,
    hot: Spectrum,
    mode_cm: float,
    coefficient_cm_per_k: float = -0.0148,
) -> dict[str, float]:
    """How much the laser heated the sample, from a band's shift.

    Compares the same mode measured at two powers. The temperature
    coefficient is a material property — the default is graphite's G band,
    about −0.015 cm⁻¹/K — and using another material's is the usual way
    this estimate goes wrong by a factor of two.

    Returns the shift, the implied temperature rise, and the coefficient
    used, so the assumption travels with the number.
    """
    if coefficient_cm_per_k == 0:
        raise ValueError("el coeficiente de temperatura no puede ser cero")
    window = 60.0
    first = cold.max_in(mode_cm - window, mode_cm + window)
    second = hot.max_in(mode_cm - window, mode_cm + window)
    if first is None or second is None:
        raise ValueError(
            f"no hay banda en {mode_cm:.0f} ± {window:.0f} cm⁻¹ en los dos "
            "espectros"
        )
    shift = float(second[0] - first[0])
    return {
        "desplazamiento_cm": shift,
        "delta_T_K": shift / coefficient_cm_per_k,
        "coeficiente_cm_por_K": coefficient_cm_per_k,
    }


def population(mode_cm: float, kelvin: float) -> float:
    """Bose–Einstein occupation of a mode. The quantity behind everything
    above: the anti-Stokes side is proportional to it and the Stokes side
    to ``n+1``."""
    if kelvin <= 0:
        return 0.0
    exponent = HC_OVER_K * abs(float(mode_cm)) / float(kelvin)
    if exponent > 700:
        return 0.0
    return 1.0 / (math.exp(exponent) - 1.0)


def expected_ratio(mode_cm: float, kelvin: float, laser_nm: float,
                   exponent: int = 3) -> float:
    """The anti-Stokes/Stokes ratio a mode should have at a temperature.

    Useful in the other direction: it says whether a mode is worth trying
    before the measurement, and it is what the round-trip test checks.
    """
    laser_cm = 1e7 / float(laser_nm)
    mode = abs(float(mode_cm))
    prefactor = ((laser_cm + mode) / (laser_cm - mode)) ** exponent
    return prefactor * math.exp(-HC_OVER_K * mode / float(kelvin))


__all__ = [
    "HC_OVER_K",
    "Temperature",
    "expected_ratio",
    "laser_heating",
    "population",
    "stokes_anti_stokes_temperature",
]
