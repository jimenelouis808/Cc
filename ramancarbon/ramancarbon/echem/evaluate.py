"""What kind of electrode this is, and how good a catalyst it is.

Two things live here that are judgements rather than measurements, and both
are the ones most often got wrong.

**Capacitor, pseudocapacitor, or battery.** The distinction is not
cosmetic. A material that stores charge through a phase transformation has
a *capacity*, and expressing it as a capacitance — dividing the charge by
the whole voltage window and quoting farads per gram — inflates the number
and describes something the material does not do. The classification here
follows the criteria of Simon, Gogotsi and Dunn: the shape of the
voltammogram, the shape of the galvanostatic curve, and the b-value. When
the verdict is battery-like, :func:`classify_storage` says so and says why,
and the modules that would quote farads carry the warning.

**A Tafel slope.** Fitted over less than a decade of current it is not
determined; fitted through a mass-transport-limited region it is a
transport property and not a kinetic one; fitted to data that were not
iR-corrected it is too large by an amount that grows with current.
:func:`tafel_analysis` refuses on the first, detects and excludes the
second, and reports the third.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .curve import ChargeDischarge, CurveError, Electrode, Voltammogram, load_echem_database
from .cv import CVResult, RateStudy
from .gcd import GCDResult

#: Minimum decades of current density for a Tafel fit to be determined.
MIN_TAFEL_DECADES = 1.0

#: Tafel slopes above this (mV/dec) usually mean transport, not kinetics.
TAFEL_TRANSPORT_LIMIT = 200.0


@dataclass
class StorageVerdict:
    """Which of the three charge-storage mechanisms this electrode uses."""

    mechanism: str
    """``"EDLC"``, ``"pseudocapacitive"`` or ``"battery"``."""
    label: str
    confidence: str
    report_as: str
    evidence: list[str] = field(default_factory=list)
    against: list[str] = field(default_factory=list)

    @property
    def farads_are_appropriate(self) -> bool:
        return self.mechanism != "battery"

    def summary(self) -> str:
        lines = [
            f"Mecanismo de almacenamiento: {self.label} "
            f"(confianza {self.confidence})",
            f"  Magnitud correcta para informar: {self.report_as}",
        ]
        if self.evidence:
            lines.append("  A favor:")
            lines.extend("    · " + e for e in self.evidence)
        if self.against:
            lines.append("  En contra:")
            lines.extend("    · " + e for e in self.against)
        if not self.farads_are_appropriate:
            lines.append("")
            lines.append(
                "  ⚠ Este electrodo NO debe informarse en F/g. Una capacitancia "
                "supone que la carga almacenada crece linealmente con el "
                "potencial, y aquí no: la carga entra a un potencial concreto, "
                "en una transformación de fase. Dividir por la ventana entera "
                "infla la cifra y describe algo que el material no hace. La "
                "magnitud es la capacidad, en C/g o mAh/g"
            )
        return "\n".join(lines)


def _battery_like_peaks(cv: CVResult) -> str:
    """Whether the voltammogram's peaks are narrow and tall, or broad humps.

    Returns a description when they are battery-like and an empty string
    otherwise. The test is the peak's FWHM against the potential window
    and its height against its own background: a phase transformation
    gives a peak a tenth of the window wide and several times the
    double-layer current, and a surface redox process gives a hump half
    the window wide and barely above it.
    """
    span = cv.curve.span
    if span <= 0.0:
        return ""
    for peak in cv.peaks:
        if peak.width_v is None:
            continue
        ratio = peak.prominence_ratio
        narrow = peak.width_v < 0.12 * span
        tall = ratio is not None and ratio > 2.0
        if narrow and tall:
            return (
                f"estrechos ({1e3 * peak.width_v:.0f} mV de anchura sobre una "
                f"ventana de {1e3 * span:.0f} mV) y {ratio:.1f} veces el fondo"
            )
    return ""


def classify_storage(
    cv: Optional[CVResult] = None,
    gcd: Optional[GCDResult] = None,
    rates: Optional[RateStudy] = None,
) -> StorageVerdict:
    """Decide the mechanism from whatever evidence is available.

    The decision is made in two steps, because the three mechanisms are
    not separated by the same evidence.

    **Battery or not** is settled by the shape: narrow, well-separated
    redox peaks; a potential plateau in the discharge; a b-value near 0.5.
    All three say the charge enters at one potential through a process
    limited by diffusion.

    **Double-layer or pseudocapacitive**, given not-battery, is settled by
    whether the voltammogram has redox features at all. The b-value cannot
    do it — both mechanisms give b near 1, which is exactly what makes
    them both "capacitive" — and a first version that treated b ≥ 0.9 as
    evidence for the double layer classified a pseudocapacitor as an
    EDLC every time, because b ≈ 1 is what a pseudocapacitor is *supposed*
    to give.

    Any combination of the three inputs works; more of them raises the
    confidence rather than changing the criteria.
    """
    data = load_echem_database()["storage_mechanisms"]
    battery = 0.0
    capacitive = 0.0
    faradaic_features = 0.0
    evidence: dict[str, list[str]] = {
        "EDLC": [], "pseudocapacitive": [], "battery": []
    }
    sources = 0

    if cv is not None:
        sources += 1
        separation = cv.peak_separation
        sharp = _battery_like_peaks(cv)
        if not cv.peaks:
            capacitive += 2.0
            evidence["EDLC"].append(
                "el voltamperograma no tiene picos redox: respuesta "
                "rectangular, propia de doble capa"
            )
        elif sharp or (separation is not None and separation > 0.15):
            battery += 2.5
            reason = []
            if sharp:
                reason.append(sharp)
            if separation is not None and separation > 0.15:
                reason.append(f"separados {1e3 * separation:.0f} mV")
            evidence["battery"].append(
                "picos redox " + " y ".join(reason)
                + ": una transformación de fase, no una reacción de superficie"
            )
        else:
            capacitive += 1.0
            faradaic_features += 2.0
            evidence["pseudocapacitive"].append(
                "hay picos redox, pero anchos y poco separados"
                + (
                    f" (ΔEp = {1e3 * separation:.0f} mV)"
                    if separation is not None
                    else ""
                )
                + ": la firma pseudocapacitiva"
            )

    if gcd is not None and gcd.discharges:
        sources += 1
        linearity = gcd.discharges[-1].linearity
        if linearity < 0.97:
            battery += 2.5
            evidence["battery"].append(
                f"la descarga tiene meseta (R² lineal = {linearity:.3f}): el "
                "potencial se queda quieto mientras entra carga, que es lo que "
                "hace una batería y no un condensador"
            )
        elif linearity >= 0.9995:
            capacitive += 2.0
            evidence["EDLC"].append(
                f"la descarga es lineal (R² = {linearity:.5f}): un triángulo"
            )
        else:
            capacitive += 2.0
            faradaic_features += 1.0
            evidence["pseudocapacitive"].append(
                f"la descarga es casi lineal pero curvada (R² = {linearity:.5f})"
            )

    if rates is not None and rates.b_values:
        sources += 1
        # The b that matters is the one at the peak, not the median over
        # the window. A battery electrode is diffusion-limited only where
        # its redox process happens; everywhere else the double layer
        # keeps b at 1, and a median over five potentials averages the
        # single informative value away.
        b = float(min(value.b for value in rates.b_values))
        if b < 0.65:
            battery += 2.0
            evidence["battery"].append(
                f"b = {b:.2f}, cerca de 0.5: corriente controlada por difusión"
            )
        elif b < 0.8:
            battery += 0.5
            capacitive += 0.5
            evidence["battery"].append(f"b = {b:.2f}: contribución difusiva apreciable")
        else:
            capacitive += 1.5
            # Deliberately NOT evidence for the double layer over
            # pseudocapacitance: both give b near 1.
            for key in ("EDLC", "pseudocapacitive"):
                evidence[key].append(
                    f"b = {b:.2f}: corriente proporcional a ν, controlada por "
                    "superficie. Esto separa capacitivo de batería, y NO "
                    "distingue doble capa de pseudocapacitivo: los dos dan b≈1"
                )

    if sources == 0:
        return StorageVerdict(
            mechanism="desconocido",
            label="no determinable",
            confidence="ninguna",
            report_as="—",
            against=["no se ha dado ninguna medida de la que deducirlo"],
        )

    if battery > capacitive:
        best = "battery"
        margin = (battery - capacitive) / max(battery + capacitive, 1e-9)
    elif faradaic_features >= 2.0:
        best = "pseudocapacitive"
        margin = faradaic_features / 4.0
    else:
        best = "EDLC"
        margin = (capacitive - battery) / max(battery + capacitive, 1e-9)

    if sources >= 3 and margin > 0.4:
        confidence = "alta"
    elif sources >= 2 and margin > 0.2:
        confidence = "media"
    else:
        confidence = "baja"

    chosen = set(evidence[best])
    against = [
        text
        for key, texts in evidence.items()
        if key != best
        for text in texts
        if text not in chosen
    ]
    return StorageVerdict(
        mechanism=best,
        label=data[best]["label"],
        confidence=confidence,
        report_as=data[best]["report_as"],
        evidence=evidence[best],
        against=against,
    )


# -- electrocatalysis --------------------------------------------------


@dataclass
class TafelResult:
    """A Tafel fit, with the conditions that make it meaningful or not."""

    slope_mv_per_decade: float
    exchange_current_density: float
    """j₀ in mA/cm², from extrapolating to zero overpotential."""
    decades: float
    r_squared: float
    range_mv: tuple[float, float]
    valid: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.valid:
            return f"Tafel: no determinable — {self.reason}"
        return (
            f"Pendiente de Tafel: {self.slope_mv_per_decade:.1f} mV/dec "
            f"({self.decades:.2f} décadas, R² = {self.r_squared:.4f}), "
            f"j₀ = {self.exchange_current_density:.3g} mA/cm²"
        )


@dataclass
class CatalysisResult:
    """HER or OER figures of merit."""

    reaction: str
    overpotential_at_benchmark: Optional[float] = None
    """η at 10 mA/cm², in volts. The field's comparison point."""
    benchmark_ma_cm2: float = 10.0
    tafel: Optional[TafelResult] = None
    onset_overpotential: Optional[float] = None
    mass_activity_a_per_g: Optional[float] = None
    ecsa_normalised: Optional[float] = None
    conversion_note: str = ""
    ir_note: str = ""
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Reacción: {self.reaction}"]
        if self.conversion_note:
            lines.append(f"  {self.conversion_note}")
        if self.ir_note:
            lines.append(f"  {self.ir_note}")
        lines.append("")
        if self.overpotential_at_benchmark is not None:
            lines.append(
                f"η a {self.benchmark_ma_cm2:g} mA/cm² : "
                f"{1e3 * self.overpotential_at_benchmark:.0f} mV"
            )
        else:
            lines.append(
                f"η a {self.benchmark_ma_cm2:g} mA/cm²: la curva no llega a esa "
                "densidad de corriente"
            )
        if self.onset_overpotential is not None:
            lines.append(
                f"η de inicio (1 mA/cm²) : "
                f"{1e3 * self.onset_overpotential:.0f} mV"
            )
        if self.tafel:
            lines.append("")
            lines.append(str(self.tafel))
            lines.extend("  ⚠ " + w for w in self.tafel.warnings)
        if self.mass_activity_a_per_g is not None:
            lines.append("")
            lines.append(
                f"Actividad másica a η del punto de referencia: "
                f"{self.mass_activity_a_per_g:.4g} A/g"
            )
        if self.ecsa_normalised is not None:
            lines.append(
                f"Corriente normalizada por ECSA: "
                f"{1e3 * self.ecsa_normalised:.4g} mA/cm² real"
            )
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


def tafel_analysis(
    overpotential: np.ndarray,
    current_density: np.ndarray,
    minimum_decades: float = MIN_TAFEL_DECADES,
) -> TafelResult:
    """Fit ``η = a + b·log₁₀|j|`` in the region where it is a straight line.

    The region is found rather than assumed: a sliding window over the
    logarithm of current density picks the longest stretch whose local
    slope is constant to within 10 %. That excludes the low-current end,
    where the reverse reaction still matters and the Tafel approximation
    does not hold, and the high-current end, where mass transport curves
    the line over — the two places a naive fit over "the linear-looking
    part" lands.

    ``current_density`` must be in mA/cm² and ``overpotential`` in volts,
    both positive.
    """
    eta = np.abs(np.asarray(overpotential, dtype=float))
    j = np.abs(np.asarray(current_density, dtype=float))
    usable = (j > 1e-6) & np.isfinite(eta) & np.isfinite(j)
    eta, j = eta[usable], j[usable]
    if eta.size < 8:
        return TafelResult(0.0, 0.0, 0.0, 0.0, (0.0, 0.0), False,
                           "hay menos de 8 puntos utilizables")

    order = np.argsort(j)
    eta, j = eta[order], j[order]
    log_j = np.log10(j)
    decades_total = float(log_j[-1] - log_j[0])
    if decades_total < minimum_decades:
        return TafelResult(
            0.0, 0.0, decades_total, 0.0, (float(1e3 * eta[0]), float(1e3 * eta[-1])),
            False,
            f"los datos sólo cubren {decades_total:.2f} décadas de corriente. "
            f"Por debajo de {minimum_decades:g} la pendiente no está "
            "determinada, y una pendiente de Tafel citada sobre menos de una "
            "década no significa nada",
        )

    best = _longest_linear(log_j, eta)
    if best is None:
        return TafelResult(
            0.0, 0.0, decades_total, 0.0, (float(1e3 * eta[0]), float(1e3 * eta[-1])),
            False,
            "no hay ningún tramo con pendiente constante: la curva no tiene "
            "región de Tafel, o el rango medido está entero en control por "
            "transporte",
        )
    start, stop = best
    window_log_j = log_j[start:stop]
    window_eta = eta[start:stop]
    slope, intercept = np.polyfit(window_log_j, window_eta, 1)
    predicted = slope * window_log_j + intercept
    total = float(((window_eta - window_eta.mean()) ** 2).sum())
    r_squared = 1.0 - float(((window_eta - predicted) ** 2).sum()) / total if total else 1.0
    exchange = float(10.0 ** (-intercept / slope)) if slope > 0 else 0.0

    result = TafelResult(
        slope_mv_per_decade=float(1e3 * slope),
        exchange_current_density=exchange,
        decades=float(window_log_j[-1] - window_log_j[0]),
        r_squared=float(r_squared),
        range_mv=(float(1e3 * window_eta[0]), float(1e3 * window_eta[-1])),
        valid=True,
    )
    if result.decades < minimum_decades:
        result.valid = False
        result.reason = (
            f"el tramo lineal sólo abarca {result.decades:.2f} décadas"
        )
    if result.slope_mv_per_decade > TAFEL_TRANSPORT_LIMIT:
        result.warnings.append(
            f"una pendiente de {result.slope_mv_per_decade:.0f} mV/dec es "
            "demasiado grande para ser cinética. Casi siempre significa una de "
            "dos cosas: falta corrección de caída óhmica, o el tramo ajustado "
            "está en control por transporte de materia. Ninguna de las dos es "
            "una propiedad del catalizador"
        )
    if stop >= eta.size - 2:
        result.warnings.append(
            "el tramo ajustado llega al extremo de mayor corriente de los "
            "datos, donde el transporte de materia empieza a curvar la recta. "
            "Comprueba que la agitación era suficiente"
        )
    return result


def _longest_linear(
    x: np.ndarray, y: np.ndarray, minimum: int = 12
) -> Optional[tuple[int, int]]:
    """The widest window in ``x`` over which ``y`` is a straight line.

    "Straight" means the least-squares residual stays within three times
    the data's own scatter — not that a local slope stays constant, which
    was the first attempt and does not work: the local gradient of a noisy
    curve fluctuates far more than the curve does, so the criterion
    fragmented a perfectly straight decade into windows a tenth of a decade
    wide and the fit was then rejected for being too short.

    The noise is estimated from the second difference, which kills any
    smooth trend, and every window's fit is evaluated in constant time from
    prefix sums, so the whole search is linear in the number of points.
    """
    n = x.size
    if n < minimum:
        return None
    second = np.diff(y, n=2)
    sigma = float(np.median(np.abs(second - np.median(second)))) * 1.4826 / math.sqrt(6.0)
    tolerance = max(3.0 * sigma, 1e-4)

    # Prefix sums, so a window's least-squares fit costs O(1).
    ones = np.arange(n + 1, dtype=float)
    sx = np.concatenate(([0.0], np.cumsum(x)))
    sy = np.concatenate(([0.0], np.cumsum(y)))
    sxx = np.concatenate(([0.0], np.cumsum(x * x)))
    sxy = np.concatenate(([0.0], np.cumsum(x * y)))
    syy = np.concatenate(([0.0], np.cumsum(y * y)))

    def rms(start: int, stop: int) -> float:
        count = ones[stop] - ones[start]
        if count < 3:
            return float("inf")
        mx = (sx[stop] - sx[start]) / count
        my = (sy[stop] - sy[start]) / count
        vxx = (sxx[stop] - sxx[start]) - count * mx * mx
        vxy = (sxy[stop] - sxy[start]) - count * mx * my
        vyy = (syy[stop] - syy[start]) - count * my * my
        if vxx <= 0.0:
            return float("inf")
        residual = max(vyy - vxy * vxy / vxx, 0.0)
        return math.sqrt(residual / count)

    best: Optional[tuple[int, int]] = None
    best_span = 0.0
    stop = minimum
    for start in range(n - minimum + 1):
        stop = max(stop, start + minimum)
        while stop < n and rms(start, stop + 1) <= tolerance:
            stop += 1
        if rms(start, stop) <= tolerance:
            span = float(x[stop - 1] - x[start])
            if span > best_span:
                best, best_span = (start, stop), span
    return best


def analyse_catalysis(
    curve: Voltammogram,
    reaction: str = "OER",
    benchmark_ma_cm2: Optional[float] = None,
    ecsa_cm2: Optional[float] = None,
) -> CatalysisResult:
    """HER or OER figures of merit from a linear sweep or a slow CV.

    Parameters
    ----------
    curve:
        The polarisation curve. Its electrode must declare the reference
        and pH so the potential can be put on the RHE scale, and its
        uncompensated resistance so the iR correction can be made.
    reaction:
        ``"HER"`` or ``"OER"``, which sets the equilibrium potential
        (0 and 1.23 V vs RHE).
    benchmark_ma_cm2:
        Current density at which the overpotential is quoted. Defaults to
        the field's 10 mA/cm², which comes from the output of a 10 %
        efficient solar cell under one sun — a useful convention, not a
        physical constant.
    ecsa_cm2:
        Electrochemically active area, from a rate study, for normalising
        the current by real rather than geometric area.

    Returns
    -------
    CatalysisResult
    """
    data = load_echem_database()
    if reaction not in ("HER", "OER"):
        raise CurveError("la reacción tiene que ser 'HER' u 'OER'")
    equilibrium = float(data["equilibrium_potentials_vs_rhe"][reaction])
    benchmark = float(
        benchmark_ma_cm2
        if benchmark_ma_cm2 is not None
        else data["benchmark_current_density_mA_cm2"]
    )
    result = CatalysisResult(reaction=reaction, benchmark_ma_cm2=benchmark)
    electrode = curve.electrode

    rhe, note = electrode.to_rhe(curve.potential)
    result.conversion_note = note
    if rhe is None:
        result.warnings.append(
            "no se puede pasar a la escala RHE, así que no hay sobrepotencial. "
            + note
        )
        return result

    corrected, ir_note = electrode.ir_correct(rhe, curve.current)
    result.ir_note = ir_note
    if not electrode.resistance_ohm:
        result.warnings.append(
            "sin corrección de caída óhmica todo lo de abajo está sesgado, y el "
            "sesgo crece con la corriente: el sobrepotencial a 10 mA/cm² sale "
            "demasiado grande y la pendiente de Tafel también"
        )

    if not electrode.area_cm2:
        result.warnings.append(
            "sin área geométrica no hay densidad de corriente, y el "
            "sobrepotencial de referencia se define sobre ella. Es el área "
            "GEOMÉTRICA la que se usa por convenio, no la electroquímica"
        )
        return result

    density = 1e3 * curve.current / electrode.area_cm2  # mA/cm2
    if reaction == "HER":
        overpotential = equilibrium - corrected
        active = density < 0
    else:
        overpotential = corrected - equilibrium
        active = density > 0
    magnitude = np.abs(density)

    if active.sum() < 8:
        result.warnings.append(
            f"apenas hay corriente en el sentido de la {reaction}. ¿Es la "
            "reacción correcta para esta curva?"
        )
        return result

    eta = overpotential[active]
    j = magnitude[active]
    order = np.argsort(j)
    eta, j = eta[order], j[order]

    if j.max() >= benchmark:
        result.overpotential_at_benchmark = float(np.interp(benchmark, j, eta))
    if j.max() >= 1.0:
        result.onset_overpotential = float(np.interp(1.0, j, eta))

    result.tafel = tafel_analysis(eta, j)

    if result.overpotential_at_benchmark is not None and electrode.mass_mg:
        current_a = benchmark * 1e-3 * electrode.area_cm2
        result.mass_activity_a_per_g = current_a / (electrode.mass_mg * 1e-3)
    if result.overpotential_at_benchmark is not None and ecsa_cm2:
        result.ecsa_normalised = benchmark * electrode.area_cm2 / ecsa_cm2

    if result.overpotential_at_benchmark is not None:
        result.warnings.append(
            f"los {benchmark:g} mA/cm² del punto de comparación son un "
            "CONVENIO (vienen del rendimiento de una celda solar del 10 % bajo "
            "1 sol), no una constante física, y se refieren al área "
            "geométrica. Comparar con la literatura exige que el otro haya "
            "usado el mismo área y la misma corrección óhmica"
        )
    result.warnings.append(
        "un sobrepotencial y una pendiente de Tafel describen la actividad en "
        "el instante de la medida. Sin una prueba de estabilidad "
        "(cronopotenciometría de horas, o mil ciclos) no dicen nada sobre si "
        "el catalizador aguanta, que es lo que decide si sirve"
    )
    return result


__all__ = [
    "MIN_TAFEL_DECADES",
    "TAFEL_TRANSPORT_LIMIT",
    "CatalysisResult",
    "StorageVerdict",
    "TafelResult",
    "analyse_catalysis",
    "classify_storage",
    "tafel_analysis",
]
