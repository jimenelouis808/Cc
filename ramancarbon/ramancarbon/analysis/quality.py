"""Is this spectrum fit to be analysed at all?

Every other module in this package assumes the data are sound and asks what
they mean. This one asks first whether they *are* sound, because the ways a
Raman measurement goes wrong are mostly invisible in the finished report:
the numbers come out, they look reasonable, and nothing says the detector
was saturated.

The checks here were chosen by breaking the analysis deliberately and seeing
what got through silently:

* **Saturation.** Clipping the top 3 % of a synthetic spectrum changed
  I_D/I_G from 1.0 to 2.0, with no warning anywhere. A saturated detector
  flattens the strongest band first, which is nearly always G, so the
  effect is to inflate every ratio measured against it.
* **Under-resolution.** The same spectrum sampled at 16 cm⁻¹ per point gave
  I_D/I_G = 3.9 instead of 1.0. Below about six points across a band the
  fit has nothing to constrain the width with.
* **Truncation.** A spectrum that stops before the band it is being asked
  about produces an absence that means nothing.
* **Calibration.** If a silicon line is in range, its distance from
  520.7 cm⁻¹ is the instrument's offset that day, and every shift the
  report interprets is measured against it.

Nothing here modifies the spectrum. It reports, in the language of what to
do about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..core.spectrum import Spectrum

#: The silicon first-order line, the usual calibration reference.
SILICON_LINE = 520.7

#: Minimum points across a band's FWHM for its width to be meaningful.
#:
#: A Lorentzian sampled at five points across its FWHM is fitted to about
#: 5 % in width; at three points the width is essentially unconstrained and
#: the fitter trades it against height without penalty.
MIN_POINTS_PER_FWHM = 6.0

#: Severity ordering used to summarise a report.
SEVERITY = ("info", "aviso", "grave")


@dataclass
class QualityIssue:
    """One thing found wrong with the measurement."""

    key: str
    severity: str
    """``"info"``, ``"aviso"`` or ``"grave"``."""
    message: str
    action: str = ""
    """What to do about it."""

    def __str__(self) -> str:
        mark = {"info": "·", "aviso": "⚠", "grave": "✗"}[self.severity]
        text = f"{mark} {self.message}"
        if self.action:
            text += f"\n    → {self.action}"
        return text


@dataclass
class QualityReport:
    """Everything the quality checks found."""

    issues: list[QualityIssue] = field(default_factory=list)
    saturated_fraction: float = 0.0
    points_per_fwhm: Optional[float] = None
    snr: Optional[float] = None
    silicon_offset: Optional[float] = None
    """Measured position of the silicon line minus 520.7 cm⁻¹, if found."""

    @property
    def worst(self) -> str:
        """The highest severity present, or ``"info"`` when nothing is wrong."""
        for level in reversed(SEVERITY):
            if any(i.severity == level for i in self.issues):
                return level
        return "info"

    @property
    def usable(self) -> bool:
        """Whether anything found is severe enough to invalidate the analysis."""
        return not any(i.severity == "grave" for i in self.issues)

    def summary(self) -> str:
        if not self.issues:
            return "Calidad del espectro: sin problemas detectados."
        lines = [f"Calidad del espectro: {self.worst}"]
        lines.extend(str(issue) for issue in self.issues)
        return "\n".join(lines)


def check_quality(
    spectrum: Spectrum,
    narrowest_fwhm_cm: Optional[float] = None,
    required_windows: Sequence[tuple[str, float, float]] = (),
) -> QualityReport:
    """Run every measurement-quality check on a raw spectrum.

    Parameters
    ----------
    spectrum:
        The **raw** spectrum, before preprocessing. Saturation is invisible
        after a baseline subtraction rescales everything, and the intensity
        histogram is only interpretable on the original counts.
    narrowest_fwhm_cm:
        Width of the narrowest band that matters, for the resolution check.
        Defaults to 30 cm⁻¹, roughly a D band in ordered material; pass the
        RBM width if you care about the RBM region.
    required_windows:
        ``(name, low, high)`` triples the analysis will need. Each one the
        spectrum does not cover is reported, so that an absence is never
        read as a result.

    Returns
    -------
    QualityReport
    """
    report = QualityReport()
    y = spectrum.intensity
    issues = report.issues

    # -- is there a spectrum at all ------------------------------------
    span = float(np.ptp(y))
    scale = max(abs(float(np.mean(y))), 1.0)
    if span <= 1e-9 * scale:
        issues.append(
            QualityIssue(
                "sin_senal",
                "grave",
                f"la intensidad no varía en todo el espectro (rango {span:.3g})",
                "No hay nada que analizar. Comprueba que el archivo no esté "
                "vacío, que no le hayas restado ya una línea base que se lo "
                "llevó todo, y que el obturador estuviera abierto",
            )
        )
        report.snr = 0.0
        return report

    # -- saturation ----------------------------------------------------
    fraction, plateau = _saturation(y)
    report.saturated_fraction = fraction
    if plateau and fraction > 0.002:
        issues.append(
            QualityIssue(
                "saturacion",
                "grave" if fraction > 0.01 else "aviso",
                f"el {fraction * 100:.1f} % de los puntos está pegado al valor "
                f"máximo ({np.max(y):.0f}): el detector se saturó",
                "Baja el tiempo de integración o la potencia y repite. La "
                "saturación aplana primero la banda más intensa, que suele ser "
                "la G, así que INFLA todos los cocientes medidos contra ella — "
                "en pruebas, recortar el 3 % superior duplicó I_D/I_G",
            )
        )

    # -- resolution ----------------------------------------------------
    narrowest = narrowest_fwhm_cm or 30.0
    step = spectrum.step
    per_fwhm = narrowest / step if step > 0 else float("inf")
    report.points_per_fwhm = per_fwhm
    if per_fwhm < MIN_POINTS_PER_FWHM:
        issues.append(
            QualityIssue(
                "resolucion",
                "grave" if per_fwhm < 3.0 else "aviso",
                f"solo hay {per_fwhm:.1f} puntos por FWHM "
                f"(paso {step:.1f} cm⁻¹, banda de {narrowest:.0f} cm⁻¹)",
                f"Hacen falta al menos {MIN_POINTS_PER_FWHM:.0f}. Por debajo de "
                "eso el ajuste no tiene con qué fijar la anchura y la "
                "intercambia con la altura: en pruebas, muestrear a 16 cm⁻¹ "
                "llevó un I_D/I_G real de 1.0 hasta 3.9. Usa una rejilla más "
                "densa o un objetivo de mayor dispersión",
            )
        )

    # -- signal to noise -----------------------------------------------
    sigma = spectrum.noise_estimate()
    if sigma > 0:
        report.snr = float(np.ptp(y) / sigma)
        if report.snr < 20.0:
            issues.append(
                QualityIssue(
                    "snr",
                    "grave" if report.snr < 8.0 else "aviso",
                    f"la relación señal/ruido del espectro completo es "
                    f"{report.snr:.0f}",
                    "Acumula más barridos: el ruido baja como la raíz del "
                    "número de acumulaciones, así que cuatro veces más tiempo "
                    "da el doble de señal/ruido. Suavizar no añade información",
                )
            )

    # -- coverage ------------------------------------------------------
    for name, low, high in required_windows:
        if not spectrum.covers(low, high, fraction=0.7):
            issues.append(
                QualityIssue(
                    f"cobertura_{name}",
                    "info",
                    f"el espectro no cubre la región {name} "
                    f"({low:.0f}–{high:.0f} cm⁻¹)",
                    "Su ausencia en el informe no significa que la banda no "
                    "esté en la muestra, solo que no se midió",
                )
            )

    # -- calibration ---------------------------------------------------
    offset = silicon_offset(spectrum)
    if offset is not None:
        report.silicon_offset = offset
        severity = "info" if abs(offset) <= 1.0 else "aviso"
        issues.append(
            QualityIssue(
                "calibracion",
                severity,
                f"hay una línea estrecha cerca de 520.7 cm⁻¹, desplazada "
                f"{offset:+.2f} cm⁻¹",
                (
                    "Si es el silicio del sustrato, ese es el error de "
                    "calibración del equipo hoy. Réstaselo a todas las "
                    "posiciones antes de interpretar desplazamientos, que son "
                    "de este mismo orden"
                    if abs(offset) > 1.0
                    else "El eje está bien calibrado si esa línea es el silicio"
                ),
            )
        )

    # -- negative excursions -------------------------------------------
    if np.any(y < 0) and float(np.min(y)) < -5.0 * max(sigma, 1e-30):
        issues.append(
            QualityIssue(
                "negativos",
                "aviso",
                f"hay intensidades negativas por debajo de 5σ "
                f"(mínimo {np.min(y):.0f})",
                "Un espectro en bruto no puede ser negativo. O ya le restaste "
                "una línea base de más, o se restó un blanco mal escalado",
            )
        )
    return report


def _saturation(y: np.ndarray) -> tuple[float, bool]:
    """Fraction of points at the maximum, and whether they form a plateau.

    A detector at full scale returns the *same* value for every saturated
    pixel, so saturation shows up as an unusually populated maximum rather
    than as a shape. Repeated maxima that are scattered across the spectrum
    are more likely a digitisation artefact than clipping, so the plateau
    test asks whether the repeats are adjacent.
    """
    if y.size < 20:
        return 0.0, False
    top = float(np.max(y))
    at_top = np.isclose(y, top, rtol=1e-9, atol=0.0)
    count = int(at_top.sum())
    if count < 3:
        return 0.0, False
    # Adjacent runs of maxima are the signature of clipping.
    runs = np.diff(np.flatnonzero(at_top))
    adjacent = int(np.count_nonzero(runs == 1))
    return count / y.size, adjacent >= 2


def silicon_offset(
    spectrum: Spectrum, window: float = 12.0, max_fwhm: float = 12.0
) -> Optional[float]:
    """Offset of a narrow line near 520.7 cm⁻¹, if there is one.

    The silicon first-order phonon is the standard calibration reference:
    narrow, strong, and at a wavelength-independent position. If the sample
    sits on a silicon substrate and the laser reaches it, the line is free
    calibration on every spectrum.

    Returns ``None`` when the region is not covered or no narrow line is
    there — a broad feature near 520 cm⁻¹ is not silicon and must not be
    used to "correct" the axis.

    Parameters
    ----------
    spectrum:
        The spectrum to look in.
    window:
        Half-width of the search, cm⁻¹.
    max_fwhm:
        Widest line still accepted as silicon. The real line is 3–5 cm⁻¹
        wide once the instrument function is included.

    Returns
    -------
    float or None
        Measured position minus 520.7 cm⁻¹.
    """
    low, high = SILICON_LINE - window, SILICON_LINE + window
    if not spectrum.covers(low, high, fraction=0.9):
        return None
    from ..core.peaks import find_peaks

    candidates = [
        p
        for p in find_peaks(spectrum, window=(low - 20.0, high + 20.0),
                            min_distance_cm=3.0, min_fwhm_cm=2.0)
        if abs(p.position - SILICON_LINE) <= window
        and p.fwhm is not None
        and p.fwhm <= max_fwhm
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda p: p.height)
    return float(best.position - SILICON_LINE)


def calibrate(spectrum: Spectrum, offset: Optional[float] = None) -> Spectrum:
    """Shift the abscissa to put a reference line where it belongs.

    Parameters
    ----------
    spectrum:
        The spectrum to correct.
    offset:
        The amount to subtract from every shift, in cm⁻¹. When omitted it
        is measured from the silicon line via :func:`silicon_offset`.

    Returns
    -------
    Spectrum
        A corrected copy, with the correction recorded in its history.

    Raises
    ------
    ValueError
        If no offset was given and no silicon line could be found — better
        than shifting the axis by a number invented from a broad feature.
    """
    value = offset if offset is not None else silicon_offset(spectrum)
    if value is None:
        raise ValueError(
            "no se ha encontrado ninguna línea estrecha cerca de 520.7 cm⁻¹ "
            "que sirva de referencia. Indica el desplazamiento a mano, o mide "
            "un patrón de silicio en la misma sesión"
        )
    out = spectrum.copy()
    out.shift = spectrum.shift - float(value)
    out.history.append(f"calibrate(offset={value:+.3f})")
    return out


__all__ = [
    "MIN_POINTS_PER_FWHM",
    "SILICON_LINE",
    "QualityIssue",
    "QualityReport",
    "calibrate",
    "check_quality",
    "silicon_offset",
]
