"""Mott-Schottky: the flat band potential and the carrier density.

For a semiconductor in contact with an electrolyte, the depletion layer
behaves as a capacitor whose thickness grows with the band bending, and
its capacitance obeys

.. math::

    \\frac{1}{C_{sc}^2} = \\frac{2}{\\varepsilon\\varepsilon_0 A^2 e N}
    \\left(E - E_{fb} - \\frac{kT}{e}\\right)

so a plot of ``1/C²`` against the DC potential is a straight line whose
**slope gives the carrier density**, whose **sign gives the type** —
positive for n-type, negative for p-type — and whose **intercept on the
potential axis gives the flat band potential**, once ``kT/e`` is taken
off it.

That is the whole method, and almost everything about it is a trap.
This module is mostly those traps, because the equation is four lines
and the reasons a Mott-Schottky number is wrong are not.

**What is measured is NOT** ``C_sc``. The cell gives the series
combination of the space-charge capacitance and the Helmholtz layer,
``1/C = 1/C_sc + 1/C_H``, and the plot is a measurement of ``C_sc``
only where ``C_sc ≪ C_H``. In depletion that holds comfortably — a
space-charge layer is tenths of a µF/cm² and a Helmholtz layer is tens
— and towards flat band it stops holding, which is precisely the end of
the plot the intercept is read from.

**And the answer depends on the frequency**, which is the single
best-known problem with the technique: on a real electrode the measured
capacitance disperses, so ``E_fb`` and ``N`` come out different at
1 kHz and at 100 Hz. A Mott-Schottky plot at one frequency is not a
measurement of anything; the check is to repeat it across the range and
report how far the answer moves. :func:`mott_schottky` takes a series
per frequency for exactly that reason, and says so when it is given
only one.

The two prefactors are the usual suspects of this package:

* ``N`` goes as ``1/A²``, so a geometric area used for a rough or
  porous electrode is wrong by that squared. Same arithmetic as
  Randles-Ševčík, same consequence.
* ``N`` goes as ``1/ε``, and ``ε`` is a tabulated number that varies
  with phase, orientation and doping. It lives in the database with its
  source and its confidence, like every other literature value here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .curve import CurveError, Impedance, load_echem_database

#: Vacuum permittivity in F/m.
EPSILON_0 = 8.8541878128e-12

#: Elementary charge in C.
ELEMENTARY_CHARGE = 1.602176634e-19

#: Boltzmann's constant in J/K.
BOLTZMANN = 1.380649e-23

#: Room temperature in K, the default when the user does not say.
ROOM_TEMPERATURE = 298.15

#: Typical Helmholtz capacitance in F/cm². The measured capacitance is
#: ``C_sc`` in SERIES with this, so the Mott-Schottky equation describes
#: the measurement only while ``C_sc`` is far below it. Twenty µF/cm² is
#: the usual order for an aqueous electrolyte; it is used to flag where
#: the assumption stops holding, never to correct anything.
HELMHOLTZ_F_PER_CM2 = 20e-6

#: Fraction of the Helmholtz capacitance above which the series
#: combination can no longer be read as ``C_sc`` alone. At 20 % the
#: error in ``1/C²`` is already 44 %.
HELMHOLTZ_SHARE = 0.20

#: How much of the measured potential window the linear region has to
#: span before the fit is worth quoting. A Mott-Schottky plot is linear
#: only in depletion; fitting the whole sweep averages that with the
#: parts where the model does not hold.
MIN_LINEAR_FRACTION = 0.30

#: Minimum number of points in the linear region.
MIN_LINEAR_POINTS = 6

#: Spread in the flat band potential across frequencies, in volts, above
#: which the dispersion is reported as the result rather than as a
#: detail. Fifty millivolts is already more than most papers' quoted
#: uncertainty.
FLAT_BAND_SPREAD_V = 0.05

#: Spread in the carrier density across frequencies, as a factor, above
#: which the same applies.
DENSITY_SPREAD_FACTOR = 2.0


@dataclass
class MottSchottkyFit:
    """One straight line through ``1/C²`` against E, at one frequency."""

    frequency_hz: float
    slope: float
    """d(1/C²)/dE in F⁻² V⁻¹, with its sign."""
    slope_error: float
    intercept: float
    """1/C² at E = 0, in F⁻²."""
    r_squared: float
    window: tuple[float, float]
    """The potential range the line was fitted over."""
    points: int
    carrier_type: str
    """``"n"`` or ``"p"``, from the SIGN of the slope."""
    carrier_density_cm3: float
    carrier_density_error_cm3: float
    flat_band_v: float
    """``E_fb`` on the measured scale, with ``kT/e`` already taken off."""
    flat_band_error_v: float
    thermal_term_v: float
    """``kT/e``. Small — 25.7 mV at 298 K — and part of the intercept:
    leaving it out shifts every flat band potential by it."""
    potentials: np.ndarray = field(default_factory=lambda: np.zeros(0))
    inverse_square: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """``1/C²`` of every point given, not only the fitted ones, so the
    figure can draw the whole sweep and shade what was used."""
    used: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        kind = {"n": "tipo n (portadores: electrones)",
                "p": "tipo p (portadores: huecos)"}[self.carrier_type]
        lines = [
            f"{self.frequency_hz:.4g} Hz — {kind}",
            f"  E_fb        = {self.flat_band_v:+.4f} "
            f"± {self.flat_band_error_v:.4f} V",
            f"  N           = {self.carrier_density_cm3:.3g} "
            f"± {self.carrier_density_error_cm3:.3g} cm⁻³",
            f"  pendiente   = {self.slope:+.4g} F⁻² V⁻¹ "
            f"(± {100 * self.slope_error / abs(self.slope):.1f} %)",
            f"  recta sobre {self.window[0]:+.3f} … {self.window[1]:+.3f} V, "
            f"{self.points} puntos, R² = {self.r_squared:.4f}",
            f"  kT/e        = {1e3 * self.thermal_term_v:.1f} mV, ya restado "
            "del corte",
        ]
        lines.extend("  ⚠ " + w for w in self.warnings)
        return "\n".join(lines)


@dataclass
class MottSchottkyResult:
    """Mott-Schottky across every frequency it was measured at."""

    fits: list[MottSchottkyFit] = field(default_factory=list)
    epsilon: float = 0.0
    epsilon_source: str = ""
    area_cm2: float = 0.0
    temperature_k: float = ROOM_TEMPERATURE
    material: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.fits)

    def flat_band_spread(self) -> float:
        """How far ``E_fb`` moves across the frequencies, in volts."""
        values = [f.flat_band_v for f in self.fits]
        return float(max(values) - min(values)) if len(values) > 1 else 0.0

    def density_spread(self) -> float:
        """Ratio of the largest carrier density to the smallest."""
        values = [f.carrier_density_cm3 for f in self.fits
                  if f.carrier_density_cm3 > 0]
        if len(values) < 2:
            return 1.0
        return float(max(values) / min(values))

    def consensus(self) -> Optional[MottSchottkyFit]:
        """The fit to quote: the one with the straightest line.

        NOT an average over frequencies. Averaging two answers that
        disagree hides the disagreement, and the disagreement is the
        result — this returns one fit and the spread is reported beside
        it.
        """
        if not self.fits:
            return None
        return max(self.fits, key=lambda f: f.r_squared)

    def summary(self) -> str:
        if not self.fits:
            return "Mott-Schottky: no hay ningún ajuste"
        lines = [
            "MOTT-SCHOTTKY",
            "─" * 46,
            f"  material      {self.material or '(sin declarar)'}",
            f"  ε             {self.epsilon:.4g}   ({self.epsilon_source})",
            f"  área          {self.area_cm2:.4g} cm²",
            f"  temperatura   {self.temperature_k:.1f} K",
            "",
        ]
        for fit in self.fits:
            lines.append(fit.summary())
            lines.append("")
        if len(self.fits) > 1:
            lines.append(
                f"  Dispersión con la frecuencia: E_fb se mueve "
                f"{1e3 * self.flat_band_spread():.0f} mV y N un factor "
                f"{self.density_spread():.2f}"
            )
            lines.append("")
        best = self.consensus()
        if best is not None:
            lines.append(
                f"  El de recta más limpia: {best.frequency_hz:.4g} Hz, "
                f"E_fb = {best.flat_band_v:+.4f} V, "
                f"N = {best.carrier_density_cm3:.3g} cm⁻³"
            )
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


def semiconductors() -> dict[str, dict]:
    """The dielectric constants on file, by key."""
    return load_echem_database().get("semiconductors", {})


def dielectric_constant(material: str) -> tuple[Optional[float], str]:
    """``(ε, where it came from)`` for a named material.

    ``(None, reason)`` when the material is not on file, which is a
    refusal and not a default: ``N`` goes as ``1/ε``, so a guessed
    permittivity is a carrier density wrong in that proportion, and the
    guess would be invisible in the result.
    """
    entry = semiconductors().get(material)
    if entry is None:
        return None, (
            f"«{material}» no está en la tabla de semiconductores de "
            "echem.json. N va como 1/ε, así que una ε supuesta es una "
            "densidad de portadores equivocada en esa proporción: añádela "
            "al JSON con su fuente, o pásala a mano"
        )
    value = entry.get("epsilon")
    if not value:
        return None, f"«{material}» no declara ε en echem.json"
    note = entry.get("note", "")
    source = (f"{entry.get('label', material)}, ε = {value:g}, "
              f"{entry.get('source', 'sin fuente')} "
              f"(confianza {entry.get('confidence', '?')})")
    return float(value), source + (f". {note}" if note else "")


def series_capacitance(spectrum: Impedance, frequency_hz: float) -> float:
    """``C_s = −1/(ω Z″)`` at one frequency, in farads.

    The **series** capacitance, which is the one Mott-Schottky is
    written for: the model is a space-charge capacitor in series with
    the Helmholtz layer, so the quantity that the equation's ``C``
    refers to is the series combination the cell presents, not ``C′``.
    The two differ by the real part of the impedance — a few per cent
    on a clean depletion response and much more once the electrode
    starts conducting — and this module uses the one the derivation
    uses.

    Interpolated in log f between measured points, and refused outside
    the measured range: an extrapolated capacitance is not a
    measurement.
    """
    low = float(spectrum.frequency.min())
    high = float(spectrum.frequency.max())
    if not low <= frequency_hz <= high:
        raise CurveError(
            f"{frequency_hz:g} Hz está fuera del barrido de "
            f"«{spectrum.name}» ({low:.4g}–{high:.4g} Hz); no se "
            "extrapola una capacitancia"
        )
    order = np.argsort(spectrum.frequency)
    frequency = spectrum.frequency[order]
    omega = 2.0 * math.pi * frequency
    imaginary = spectrum.z.imag[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        series = np.where(imaginary != 0.0, -1.0 / (omega * imaginary), np.nan)
    value = float(np.interp(math.log10(frequency_hz),
                            np.log10(frequency), series))
    if not math.isfinite(value) or value <= 0.0:
        raise CurveError(
            f"la capacitancia en serie a {frequency_hz:g} Hz sale "
            f"{value:g} F en «{spectrum.name}»: ahí Z″ pasa por cero o es "
            "positivo, así que el electrodo no es capacitivo a esa "
            "frecuencia y no hay un punto de Mott-Schottky que poner"
        )
    return value


def series_from_spectra(
    spectra: Sequence[Impedance], frequency_hz: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``(potentials, capacitances, notes)`` from a set of EIS spectra.

    The DC bias of each spectrum is read from its metadata, which the
    EC-Lab reader fills from the file's own ``<Ewe>/V`` column. A folder
    of PEIS files measured at different biases IS a Mott-Schottky
    experiment; the only thing it was missing was which potential each
    file belongs to, and typing those back in by hand is how a point
    ends up on the wrong abscissa.
    """
    potentials: list[float] = []
    capacitances: list[float] = []
    notes: list[str] = []
    for spectrum in spectra:
        bias = spectrum.metadata.get("potencial_cd_v")
        if bias is None:
            notes.append(
                f"«{spectrum.name}» no declara el potencial de polarización, "
                "así que no puede ser un punto de Mott-Schottky. Si viene de "
                "un .mpr, la columna <Ewe>/V lo lleva"
            )
            continue
        try:
            capacitances.append(series_capacitance(spectrum, frequency_hz))
        except CurveError as exc:
            notes.append(str(exc))
            continue
        potentials.append(float(bias))
    order = np.argsort(potentials) if potentials else np.zeros(0, dtype=int)
    return (np.asarray(potentials)[order],
            np.asarray(capacitances)[order], notes)


def _fit_window(potential: np.ndarray, inverse: np.ndarray
                ) -> Optional[tuple[int, int]]:
    """The widest straight stretch of ``1/C²`` against E.

    Found, not assumed. A Mott-Schottky plot is linear only in the
    depletion region: outside it the curve bends, towards flat band
    because the Helmholtz layer stops being negligible and at the far
    end because the electrode starts to conduct. Fitting the whole sweep
    returns the slope of a line through points that are not on one, and
    nothing in the result says so.

    This is the same search the Tafel region uses, for the same reason
    and with the same argument against a local-slope criterion.
    """
    from .evaluate import _longest_linear

    return _longest_linear(potential, inverse, minimum=MIN_LINEAR_POINTS)


def _line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """``(slope, intercept, slope error, R²)`` of a straight line."""
    count = x.size
    mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
    vxx = float(np.sum((x - mean_x) ** 2))
    vxy = float(np.sum((x - mean_x) * (y - mean_y)))
    vyy = float(np.sum((y - mean_y) ** 2))
    if vxx <= 0.0:
        raise CurveError("todos los potenciales son el mismo")
    slope = vxy / vxx
    intercept = mean_y - slope * mean_x
    residual = max(vyy - slope * vxy, 0.0)
    if count > 2:
        error = math.sqrt(residual / (count - 2) / vxx)
    else:
        error = float("inf")
    r_squared = 1.0 - residual / vyy if vyy > 0 else 1.0
    return slope, intercept, error, r_squared


def mott_schottky(
    potential: np.ndarray,
    capacitance: np.ndarray,
    frequency_hz: float,
    epsilon: float,
    area_cm2: float,
    temperature_k: float = ROOM_TEMPERATURE,
) -> MottSchottkyFit:
    """One Mott-Schottky line, at one frequency.

    ``capacitance`` in farads, ``potential`` in volts on whatever scale
    the electrode was measured against — the flat band potential comes
    back on that same scale, because converting it would need the pH and
    a conversion is a claim.
    """
    potential = np.asarray(potential, dtype=float)
    capacitance = np.asarray(capacitance, dtype=float)
    if potential.size != capacitance.size:
        raise CurveError("potencial y capacitancia no tienen igual longitud")
    if potential.size < MIN_LINEAR_POINTS:
        raise CurveError(
            f"hacen falta al menos {MIN_LINEAR_POINTS} potenciales para un "
            f"Mott-Schottky; hay {potential.size}. Una recta por tres puntos "
            "siempre ajusta"
        )
    if np.any(capacitance <= 0.0):
        raise CurveError("hay capacitancias nulas o negativas")
    if epsilon <= 0.0 or area_cm2 <= 0.0:
        raise CurveError("ε y el área tienen que ser positivos")

    order = np.argsort(potential)
    potential, capacitance = potential[order], capacitance[order]
    inverse = 1.0 / capacitance ** 2

    window = _fit_window(potential, inverse)
    warnings: list[str] = []
    if window is None:
        start, stop = 0, potential.size
        warnings.append(
            "no se encontró ninguna región recta, así que la recta va por "
            "TODOS los puntos. Un Mott-Schottky es lineal sólo en la zona "
            "de agotamiento: una pendiente ajustada sobre un gráfico curvo "
            "no mide ninguna densidad de portadores"
        )
    else:
        start, stop = window
    chosen = slice(start, stop)
    slope, intercept, slope_error, r_squared = _line(
        potential[chosen], inverse[chosen])

    span = float(potential[stop - 1] - potential[start])
    whole = float(potential[-1] - potential[0])
    if whole > 0 and span / whole < MIN_LINEAR_FRACTION:
        warnings.append(
            f"la región recta mide {span:.3f} V de los {whole:.3f} V "
            f"barridos ({100 * span / whole:.0f} %). Eso es poco para una "
            "recta: mira la figura antes de citar la pendiente"
        )

    thermal = BOLTZMANN * temperature_k / ELEMENTARY_CHARGE
    carrier_type = "n" if slope > 0 else "p"
    # E_fb from the intercept on the potential axis, with kT/e removed.
    # The sign of that correction follows the type, which is the sign of
    # the slope, so it is read off the fit rather than assumed.
    crossing = -intercept / slope
    flat_band = crossing - thermal if carrier_type == "n" else crossing + thermal
    # Propagated from the slope and the intercept through E = -b/m. The
    # intercept's own error is the slope's scaled by the mean abscissa,
    # which is what makes a plot far from E = 0 give a worse E_fb than
    # one that straddles it.
    mean_x = float(np.mean(potential[chosen]))
    crossing_error = abs(slope_error / slope) * abs(crossing - mean_x) \
        if slope else float("inf")

    area_m2 = area_cm2 * 1e-4
    prefactor = 2.0 / (epsilon * EPSILON_0 * area_m2 ** 2 * ELEMENTARY_CHARGE)
    density_m3 = prefactor / abs(slope)
    density_cm3 = density_m3 * 1e-6
    density_error = density_cm3 * abs(slope_error / slope) if slope else 0.0

    # The measured capacitance is C_sc in SERIES with the Helmholtz
    # layer, so the plot measures C_sc only while C_sc is far below it.
    ceiling = HELMHOLTZ_SHARE * HELMHOLTZ_F_PER_CM2 * area_cm2
    biggest = float(capacitance[chosen].max())
    if biggest > ceiling:
        warnings.append(
            f"la capacitancia llega a {1e6 * biggest / area_cm2:.2f} µF/cm², "
            f"que es más del {100 * HELMHOLTZ_SHARE:.0f} % de una capa de "
            f"Helmholtz típica ({1e6 * HELMHOLTZ_F_PER_CM2:.0f} µF/cm²). Lo "
            "que se mide es C_sc EN SERIE con ella, así que ahí 1/C² ya no "
            "es 1/C_sc²: el gráfico se curva hacia el potencial de banda "
            "plana por esto y no por la muestra"
        )
    if r_squared < 0.98:
        warnings.append(
            f"R² = {r_squared:.4f} sobre la región elegida. Un "
            "Mott-Schottky que no sale recto no da ni E_fb ni N"
        )

    used = np.zeros(potential.size, dtype=bool)
    used[chosen] = True
    return MottSchottkyFit(
        frequency_hz=float(frequency_hz), slope=slope,
        slope_error=slope_error, intercept=intercept, r_squared=r_squared,
        window=(float(potential[start]), float(potential[stop - 1])),
        points=int(stop - start), carrier_type=carrier_type,
        carrier_density_cm3=density_cm3,
        carrier_density_error_cm3=density_error,
        flat_band_v=flat_band, flat_band_error_v=crossing_error,
        thermal_term_v=thermal, potentials=potential,
        inverse_square=inverse, used=used, warnings=warnings,
    )


def analyse_mott_schottky(
    spectra: Sequence[Impedance],
    frequencies: Sequence[float],
    epsilon: Optional[float] = None,
    material: str = "",
    area_cm2: Optional[float] = None,
    temperature_k: float = ROOM_TEMPERATURE,
) -> MottSchottkyResult:
    """Mott-Schottky from a set of EIS spectra, at several frequencies.

    ``spectra`` are impedance spectra of the SAME electrode at different
    DC biases; each one's bias is read from its metadata. ``frequencies``
    is where to read the capacitance — more than one on purpose, because
    a Mott-Schottky at a single frequency is not a measurement and the
    only way to find out is to do it at several and look at how far the
    answer moves.
    """
    result = MottSchottkyResult(material=material,
                                temperature_k=temperature_k)
    if epsilon is None:
        if not material:
            result.warnings.append(
                "no hay ni ε ni material: sin la constante dieléctrica no "
                "se puede dar una densidad de portadores, porque N va como "
                "1/ε. El potencial de banda plana NO la necesita"
            )
            return result
        epsilon, source = dielectric_constant(material)
        if epsilon is None:
            result.warnings.append(source)
            return result
        result.epsilon_source = source
    else:
        result.epsilon_source = "ε dada a mano"
    result.epsilon = float(epsilon)

    electrode = spectra[0].electrode if spectra else None
    if area_cm2 is None:
        area_cm2 = getattr(electrode, "area_cm2", None)
    if not area_cm2:
        result.warnings.append(
            "no hay área: N va como 1/A², así que no se puede dar una "
            "densidad de portadores. El potencial de banda plana NO la "
            "necesita, y sale igual"
        )
        area_cm2 = 1.0
        result.warnings.append(
            "se ha usado 1 cm² para poder dibujar; la N que salga está "
            "escalada por (A/1 cm²)² y no es una medida"
        )
    result.area_cm2 = float(area_cm2)

    for frequency in frequencies:
        potential, capacitance, notes = series_from_spectra(spectra, frequency)
        result.warnings.extend(notes)
        if potential.size < MIN_LINEAR_POINTS:
            result.warnings.append(
                f"a {frequency:g} Hz sólo quedan {potential.size} puntos; "
                f"hacen falta {MIN_LINEAR_POINTS}"
            )
            continue
        try:
            fit = mott_schottky(potential, capacitance, frequency,
                                result.epsilon, result.area_cm2,
                                temperature_k)
        except CurveError as exc:
            result.warnings.append(f"a {frequency:g} Hz: {exc}")
            continue
        result.fits.append(fit)

    _check_the_frequencies(result)
    return result


def _check_the_frequencies(result: MottSchottkyResult) -> None:
    """The dispersion with frequency, which is the known failure mode.

    On a real electrode the capacitance disperses — surface states,
    roughness, a non-ideal interface — so ``E_fb`` and ``N`` come out
    different at every frequency. A paper that quotes one frequency and
    does not say which is not reporting a measurement, and the way to
    find out is this: measure at several and look at the spread.
    """
    if not result.fits:
        return
    if len(result.fits) == 1:
        result.warnings.append(
            "sólo se ha hecho a UNA frecuencia. El resultado de un "
            "Mott-Schottky depende de la frecuencia elegida y ése es su "
            "problema conocido: en un electrodo real la capacitancia "
            "dispersa, así que E_fb y N salen distintos a 1 kHz y a 100 Hz. "
            "Repítelo en varias y mira cuánto se mueve la respuesta — es la "
            "única forma de saber si lo que sale significa algo"
        )
        return

    spread = result.flat_band_spread()
    if spread > FLAT_BAND_SPREAD_V:
        values = ", ".join(f"{f.frequency_hz:.4g} Hz → {f.flat_band_v:+.3f} V"
                           for f in result.fits)
        result.warnings.append(
            f"E_fb se mueve {1e3 * spread:.0f} mV entre frecuencias "
            f"({values}). ESO es el resultado: la capacitancia dispersa, "
            "así que el electrodo no es la unión ideal que el modelo supone. "
            "Cita el intervalo, no una de las cifras"
        )
    factor = result.density_spread()
    if factor > DENSITY_SPREAD_FACTOR:
        result.warnings.append(
            f"N cambia un factor {factor:.1f} entre frecuencias. Por la "
            "misma razón: lo que dispersa es la capacitancia, y N sale de "
            "su pendiente"
        )
    types = {fit.carrier_type for fit in result.fits}
    if len(types) > 1:
        result.warnings.append(
            "la PENDIENTE cambia de signo entre frecuencias, así que el "
            "tipo de portador sale n en unas y p en otras. Eso no es un "
            "semiconductor ambipolar: es que la recta no está midiendo una "
            "capa de agotamiento en al menos una de ellas"
        )


__all__ = [
    "BOLTZMANN",
    "DENSITY_SPREAD_FACTOR",
    "ELEMENTARY_CHARGE",
    "EPSILON_0",
    "FLAT_BAND_SPREAD_V",
    "HELMHOLTZ_F_PER_CM2",
    "HELMHOLTZ_SHARE",
    "MIN_LINEAR_FRACTION",
    "MIN_LINEAR_POINTS",
    "MottSchottkyFit",
    "MottSchottkyResult",
    "ROOM_TEMPERATURE",
    "analyse_mott_schottky",
    "dielectric_constant",
    "mott_schottky",
    "semiconductors",
    "series_capacitance",
    "series_from_spectra",
]
