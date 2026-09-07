"""Cyclic voltammetry: capacitance, redox peaks, kinetics.

The three numbers people take from a voltammogram are a capacitance, a
peak separation, and a b-value, and each has a trap in it.

**Capacitance depends on a convention nobody states.** Integrating the
closed loop and dividing by ``2 ν ΔV`` gives one answer; integrating the
anodic sweep alone and dividing by ``ν ΔV`` gives the same answer; and
integrating the closed loop and dividing by ``ν ΔV`` — which is also
published — gives exactly twice it. Both defensible results are computed
here and both are labelled, because a capacitance quoted without its
convention is a number with a factor-of-two uncertainty.

**A peak separation is only a kinetic quantity after iR correction.**
Uncompensated resistance widens ΔEp in proportion to the current, so a
"quasi-reversible" system is often a reversible one measured through
30 Ω. The value is reported with the correction status attached.

**The b-value needs a decade of scan rates, not three points.** ``i = a ν^b``
is fitted in logarithms, and the fit is reported with the span of the data
that produced it; over a factor of three in ν, b is not determined well
enough to distinguish 0.8 from 1.0, which is the only distinction anyone
uses it for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.signal import find_peaks as _scipy_find_peaks
from scipy.signal import peak_widths as _peak_widths

from ..core.compat import trapezoid
from .curve import CurveError, Electrode, Voltammogram, load_echem_database

#: Minimum span in scan rate, as a ratio, for a b-value to mean anything.
MIN_RATE_SPAN = 8.0

#: Fraction of the potential window trimmed at each end before integrating.
#: The switching transient at a vertex is instrument response, not sample
#: current, and including it inflates the capacitance of a fast scan.
VERTEX_TRIM = 0.02

#: Minimum prominence of a redox peak, relative to the mean |current|.
PEAK_PROMINENCE = 0.08


@dataclass
class Capacitance:
    """A capacitance, with the convention that produced it."""

    farads: float
    convention: str
    specific_f_per_g: Optional[float] = None
    specific_f_per_cm2: Optional[float] = None
    specific_f_per_cm3: Optional[float] = None
    note: str = ""

    def __str__(self) -> str:
        parts = [f"{self.farads * 1e3:.4g} mF ({self.convention})"]
        if self.specific_f_per_g is not None:
            parts.append(f"{self.specific_f_per_g:.4g} F/g")
        if self.specific_f_per_cm2 is not None:
            parts.append(f"{self.specific_f_per_cm2 * 1e3:.4g} mF/cm²")
        if self.specific_f_per_cm3 is not None:
            parts.append(f"{self.specific_f_per_cm3:.4g} F/cm³")
        return "  ".join(parts)


@dataclass
class RedoxPeak:
    """One peak of a voltammogram."""

    potential: float
    current: float
    branch: str
    """``"anódico"`` or ``"catódico"``."""
    prominence: float
    width_v: Optional[float] = None
    """FWHM in volts. Together with the prominence this is what separates a
    battery-like peak from a pseudocapacitive hump: the first is narrow and
    tall against its background, the second broad and only a little above
    it. The peak separation alone does not do it — plenty of battery
    materials have ΔEp below 100 mV."""
    baseline: float = 0.0
    """Current just outside the peak, for the prominence ratio."""

    @property
    def prominence_ratio(self) -> Optional[float]:
        if self.baseline <= 0.0:
            return None
        return self.prominence / self.baseline

    def __str__(self) -> str:
        text = (
            f"{self.branch:<10s} E = {self.potential:+.4f} V   "
            f"i = {1e3 * self.current:+.4g} mA"
        )
        if self.width_v:
            text += f"   FWHM {1e3 * self.width_v:.0f} mV"
        ratio = self.prominence_ratio
        if ratio is not None:
            text += f"   pico/fondo {ratio:.1f}"
        return text


@dataclass
class CVResult:
    """The analysis of one voltammogram."""

    curve: Voltammogram
    charge_c: float = 0.0
    """Anodic charge in C, the integral of the positive-current branch."""
    capacitance_loop: Optional[Capacitance] = None
    capacitance_sweep: Optional[Capacitance] = None
    capacity_c_per_g: Optional[float] = None
    capacity_mah_per_g: Optional[float] = None
    peaks: list[RedoxPeak] = field(default_factory=list)
    peak_separation: Optional[float] = None
    formal_potential: Optional[float] = None
    reversibility: str = ""
    ir_note: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def peak_pairs(self) -> list[tuple[RedoxPeak, RedoxPeak]]:
        anodic = [p for p in self.peaks if p.branch == "anódico"]
        cathodic = [p for p in self.peaks if p.branch == "catódico"]
        pairs = []
        for a in anodic:
            if not cathodic:
                break
            partner = min(cathodic, key=lambda c: abs(c.potential - a.potential))
            pairs.append((a, partner))
        return pairs

    def summary(self) -> str:
        lines = [self.curve.describe(), self.curve.electrode.describe(), ""]
        lines.append(f"Carga anódica: {self.charge_c * 1e3:.4g} mC")
        if self.capacitance_loop:
            lines.append("")
            lines.append("Capacitancia:")
            lines.append("  " + str(self.capacitance_loop))
            if self.capacitance_sweep:
                lines.append("  " + str(self.capacitance_sweep))
            lines.append(
                "  Las dos son la MISMA medida en convenios distintos. El de "
                "lazo integra el ciclo cerrado y divide por 2νΔV; el de rama "
                "integra solo la anódica y divide por νΔV. Publicar sin decir "
                "cuál deja un factor 2 de ambigüedad"
            )
        if self.capacity_c_per_g is not None:
            lines.append("")
            lines.append(
                f"Capacidad: {self.capacity_c_per_g:.4g} C/g = "
                f"{self.capacity_mah_per_g:.4g} mAh/g"
            )
        if self.peaks:
            lines.append("")
            lines.append("Picos redox:")
            lines.extend("  " + str(p) for p in self.peaks)
            if self.peak_separation is not None:
                lines.append(
                    f"  ΔEp = {1e3 * self.peak_separation:.1f} mV, "
                    f"E°' = {self.formal_potential:+.4f} V — {self.reversibility}"
                )
        if self.ir_note:
            lines.append("")
            lines.append(self.ir_note)
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


def integrate_charge(
    curve: Voltammogram, trim: float = VERTEX_TRIM
) -> tuple[float, float, float]:
    """``(anodic charge, absolute loop charge, integrated span)`` in C and V.

    ``q = ∫ i dt = ∫ i dE / ν``. The ends of the window are trimmed by
    ``trim`` of the span: at a vertex the potentiostat reverses in a finite
    time and the current spike there is instrument response, not sample
    current, and it inflates the capacitance of a fast scan by several per
    cent.
    """
    low, high = curve.window
    margin = trim * (high - low)
    inside = (curve.potential >= low + margin) & (curve.potential <= high - margin)
    if inside.sum() < 10:
        inside = np.ones(curve.n, dtype=bool)

    potential = curve.potential[inside]
    current = curve.current[inside]
    time = np.abs(np.concatenate(([0.0], np.cumsum(np.abs(np.diff(potential))))))
    time = time / curve.scan_rate

    anodic = np.where(current > 0.0, current, 0.0)
    # The span actually integrated, which is what the charge must be
    # divided by. Trimming the vertices and then dividing by the FULL
    # window makes every capacitance 2 x trim too small — 4 % with the
    # default, a systematic error that looks like a real difference
    # between samples measured with different vertex settings.
    span = float(potential.max() - potential.min())
    return (
        float(trapezoid(anodic, time)),
        float(trapezoid(np.abs(current), time)),
        span,
    )


def capacitance(
    curve: Voltammogram, window: Optional[tuple[float, float]] = None
) -> tuple[Capacitance, Capacitance]:
    """Capacitance from a voltammogram, in both usual conventions.

    Returns ``(loop, sweep)``. Numerically the two agree; they are computed
    separately so that a mismatch — which happens when the loop is not
    closed, or when one branch has a faradaic contribution the other lacks
    — is visible instead of averaged away.
    """
    _, absolute, integrated_span = integrate_charge(curve)
    if window is not None:
        low, high = window
        span = high - low
    else:
        span = integrated_span
    if span <= 0.0:
        raise CurveError("la ventana de potencial es nula")
    loop_value = absolute / (2.0 * span)

    try:
        anodic_sweep, _ = curve.sweeps()
        anodic_charge, _, sweep_span = integrate_charge(anodic_sweep)
        sweep_value = anodic_charge / (span if window is not None else sweep_span)
    except CurveError:
        sweep_value = loop_value

    electrode = curve.electrode
    results = []
    for value, convention in (
        (loop_value, "lazo cerrado ÷ 2νΔV"),
        (sweep_value, "rama anódica ÷ νΔV"),
    ):
        results.append(
            Capacitance(
                farads=value,
                convention=convention,
                specific_f_per_g=electrode.specific(value, "mass"),
                specific_f_per_cm2=electrode.specific(value, "area"),
                specific_f_per_cm3=electrode.specific(value, "volume"),
            )
        )
    return results[0], results[1]


def find_redox_peaks(
    curve: Voltammogram, prominence: float = PEAK_PROMINENCE
) -> list[RedoxPeak]:
    """Locate anodic and cathodic peaks.

    Peaks are sought on each sweep separately, because a voltammogram
    traversed as one array doubles back on itself and a peak finder run on
    the concatenation reports the turning points of the sweep as peaks of
    the sample.
    """
    try:
        anodic, cathodic = curve.sweeps()
    except CurveError:
        return []
    scale = float(np.abs(curve.current).mean())
    if scale <= 0.0:
        return []

    peaks: list[RedoxPeak] = []
    for sweep, branch, sign in ((anodic, "anódico", 1.0), (cathodic, "catódico", -1.0)):
        order = np.argsort(sweep.potential)
        potential = sweep.potential[order]
        current = sign * sweep.current[order]
        indices, properties = _scipy_find_peaks(
            current, prominence=prominence * scale
        )
        if indices.size == 0:
            continue
        widths = _peak_widths(current, indices, rel_height=0.5)[0]
        step = float(np.median(np.diff(potential))) if potential.size > 1 else 0.0
        for index, value, width in zip(indices, properties["prominences"], widths):
            baseline = float(current[index]) - float(value)
            peaks.append(
                RedoxPeak(
                    potential=float(potential[index]),
                    current=float(sign * current[index]),
                    branch=branch,
                    prominence=float(value),
                    width_v=float(width * step) if width > 0 and step > 0 else None,
                    baseline=abs(baseline),
                )
            )
    peaks.sort(key=lambda p: (p.branch, p.potential))
    return peaks


def analyse_cv(
    curve: Voltammogram,
    use_cycle: int = -1,
    window: Optional[tuple[float, float]] = None,
) -> CVResult:
    """Full analysis of one voltammogram.

    Parameters
    ----------
    curve:
        The measurement. If it contains several cycles they are split and
        one is used.
    use_cycle:
        Which cycle, ``-1`` for the last. The **first cycle is not
        representative**: it carries irreversible formation, electrolyte
        decomposition or oxide conditioning, and averaging it with the rest
        mixes a one-off process into a steady-state number.
    window:
        Potential window for the capacitance, defaulting to the measured
        one.

    Returns
    -------
    CVResult
    """
    cycles = curve.cycles()
    chosen = cycles[use_cycle] if len(cycles) > 1 else curve
    result = CVResult(curve=chosen)
    if len(cycles) > 1:
        result.warnings.append(
            f"el archivo tiene {len(cycles)} ciclos; se ha usado el "
            f"{'último' if use_cycle == -1 else str(use_cycle)}. El PRIMERO "
            "nunca es representativo: lleva formación irreversible, "
            "descomposición de electrolito o acondicionamiento del óxido"
        )

    corrected, note = chosen.electrode.ir_correct(chosen.potential, chosen.current)
    result.ir_note = note
    working = Voltammogram(
        potential=corrected,
        current=chosen.current,
        scan_rate=chosen.scan_rate,
        electrode=chosen.electrode,
        name=chosen.name,
        metadata=chosen.metadata,
    )

    anodic_charge, _, _ = integrate_charge(working)
    result.charge_c = anodic_charge
    result.capacitance_loop, result.capacitance_sweep = capacitance(working, window)

    per_gram = chosen.electrode.specific(anodic_charge, "mass")
    if per_gram is not None:
        result.capacity_c_per_g = per_gram
        result.capacity_mah_per_g = per_gram / 3.6

    result.peaks = find_redox_peaks(working)
    pairs = result.peak_pairs
    if pairs:
        anodic, cathodic = max(pairs, key=lambda p: abs(p[0].current))
        result.peak_separation = abs(anodic.potential - cathodic.potential)
        result.formal_potential = 0.5 * (anodic.potential + cathodic.potential)
        result.reversibility = _reversibility(
            result.peak_separation, bool(chosen.electrode.resistance_ohm)
        )

    _cv_warnings(result, chosen)
    return result


def _reversibility(separation: float, ir_corrected: bool) -> str:
    """Read ΔEp, with the caveat it needs."""
    millivolts = 1e3 * separation
    if millivolts < 45.0:
        verdict = (
            "ΔEp por debajo de los 59 mV de un proceso reversible de un "
            "electrón: o intervienen más electrones, o el par es de superficie "
            "(adsorbido), donde ΔEp tiende a cero"
        )
    elif millivolts < 80.0:
        verdict = "compatible con un proceso reversible de un electrón (59 mV)"
    elif millivolts < 200.0:
        verdict = "cuasi-reversible"
    else:
        verdict = "irreversible, o muy limitado por transferencia de carga"
    if not ir_corrected:
        verdict += (
            ". OJO: sin corrección de caída óhmica, ΔEp crece en proporción a "
            "la corriente, y un sistema reversible medido a través de 30 Ω "
            "parece cuasi-reversible"
        )
    return verdict


def _cv_warnings(result: CVResult, curve: Voltammogram) -> None:
    electrode = curve.electrode
    if electrode.mass_mg is None:
        result.warnings.append(
            "sin masa activa no hay capacitancia específica. Y tiene que ser "
            "la masa de material ACTIVO, no la del electrodo: el aglomerante "
            "y el carbón conductor suelen ser el 20 %, y usar la masa total "
            "rebaja el resultado justo esa fracción"
        )
    if curve.scan_rate < 0.002:
        result.warnings.append(
            f"a {1e3 * curve.scan_rate:g} mV/s la medida dura mucho y la "
            "corriente es pequeña: comprueba que no hay deriva del electrodo "
            "de referencia ni autodescarga contribuyendo a la integral"
        )
    if result.peaks:
        result.warnings.append(
            "hay picos redox definidos. Antes de citar una capacitancia en "
            "F/g, mira el mecanismo (echem.evaluate): un material con picos "
            "almacena CARGA, y expresarlo en faradios dividiendo por la "
            "ventana infla el número y no describe lo que hace el material"
        )


# -- kinetics from a series of scan rates ------------------------------


@dataclass
class BValue:
    """The exponent of ``i = a ν^b`` at one potential."""

    potential: float
    b: float
    b_error: float
    a: float
    rates: tuple[float, ...]
    r_squared: float

    @property
    def interpretation(self) -> str:
        if self.b >= 0.9:
            return "controlado por superficie (capacitivo)"
        if self.b >= 0.75:
            return "mixto, con predominio capacitivo"
        if self.b >= 0.6:
            return "mixto"
        return "controlado por difusión (tipo batería)"

    def __str__(self) -> str:
        return (
            f"E = {self.potential:+.3f} V:  b = {self.b:.3f} ± {self.b_error:.3f}  "
            f"(R² = {self.r_squared:.4f}) — {self.interpretation}"
        )


@dataclass
class RateStudy:
    """What a set of voltammograms at different scan rates says."""

    rates: list[float] = field(default_factory=list)
    capacitances: list[float] = field(default_factory=list)
    b_values: list[BValue] = field(default_factory=list)
    capacitive_fraction: Optional[dict[float, float]] = None
    """Scan rate → fraction of the current that is surface-controlled."""
    cdl_farads: Optional[float] = None
    ecsa_cm2: Optional[float] = None
    ecsa_range_cm2: Optional[tuple[float, float]] = None
    trasatti_total: Optional[float] = None
    trasatti_outer: Optional[float] = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Estudio de velocidad de barrido "
            f"({len(self.rates)} velocidades, "
            f"{1e3 * min(self.rates):g}–{1e3 * max(self.rates):g} mV/s)"
        ]
        if self.capacitances:
            lines.append("")
            lines.append("Capacitancia frente a velocidad:")
            for rate, value in zip(self.rates, self.capacitances):
                lines.append(f"  {1e3 * rate:8.1f} mV/s   {1e3 * value:10.4g} mF")
            retention = 100.0 * self.capacitances[-1] / max(self.capacitances[0], 1e-30)
            lines.append(
                f"  Retención de la más lenta a la más rápida: {retention:.1f} %"
            )
        if self.b_values:
            lines.append("")
            lines.append("Análisis de b (i = a·ν^b):")
            lines.extend("  " + str(b) for b in self.b_values)
        if self.capacitive_fraction:
            lines.append("")
            lines.append("Reparto capacitivo/difusivo (Dunn, i = k₁ν + k₂√ν):")
            for rate, fraction in sorted(self.capacitive_fraction.items()):
                lines.append(
                    f"  {1e3 * rate:8.1f} mV/s   capacitivo {100 * fraction:5.1f} %"
                )
        if self.trasatti_total is not None:
            lines.append("")
            lines.append("Trasatti:")
            lines.append(
                f"  capacitancia total (ν→0)  : {1e3 * self.trasatti_total:.4g} mF"
            )
            lines.append(
                f"  superficie externa (ν→∞)  : {1e3 * self.trasatti_outer:.4g} mF"
            )
            if self.trasatti_total > 0:
                lines.append(
                    f"  accesible sólo despacio    : "
                    f"{100 * (1 - self.trasatti_outer / self.trasatti_total):.1f} %"
                )
        if self.cdl_farads is not None:
            lines.append("")
            lines.append(f"C_dl = {1e6 * self.cdl_farads:.4g} µF")
            if self.ecsa_cm2 is not None:
                low, high = self.ecsa_range_cm2 or (self.ecsa_cm2, self.ecsa_cm2)
                lines.append(
                    f"ECSA = {self.ecsa_cm2:.4g} cm² (entre {low:.3g} y "
                    f"{high:.3g} según la Cs que se adopte)"
                )
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """``(slope, intercept, slope error, R²)`` by ordinary least squares."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        return 0.0, 0.0, float("inf"), 0.0
    mean_x, mean_y = x.mean(), y.mean()
    sxx = float(((x - mean_x) ** 2).sum())
    if sxx <= 0.0:
        return 0.0, float(mean_y), float("inf"), 0.0
    slope = float(((x - mean_x) * (y - mean_y)).sum() / sxx)
    intercept = float(mean_y - slope * mean_x)
    residual = y - (slope * x + intercept)
    total = float(((y - mean_y) ** 2).sum())
    r_squared = 1.0 - float((residual**2).sum()) / total if total > 0 else 1.0
    if n > 2:
        variance = float((residual**2).sum()) / (n - 2)
        error = math.sqrt(variance / sxx)
    else:
        error = float("inf")
    return slope, intercept, error, r_squared


def analyse_rate_study(
    curves: Sequence[Voltammogram],
    b_potentials: Optional[Sequence[float]] = None,
    ecsa_material: str = "por_defecto",
    non_faradaic: bool = False,
) -> RateStudy:
    """Everything a series of voltammograms at different rates yields.

    Parameters
    ----------
    curves:
        Two or more voltammograms of the same electrode, differing only in
        scan rate.
    b_potentials:
        Potentials at which to fit ``i = a ν^b``. Defaults to five points
        spread across the common window.
    ecsa_material:
        Key in ``echem.json`` for the specific capacitance used to turn
        C_dl into an area.
    non_faradaic:
        Set when the window was chosen deliberately free of faradaic
        current, which is what makes the C_dl and the ECSA meaningful. When
        it is not set they are still computed, and flagged.

    Returns
    -------
    RateStudy
    """
    if len(curves) < 2:
        raise CurveError("hacen falta al menos dos velocidades de barrido")
    ordered = sorted(curves, key=lambda c: c.scan_rate)
    study = RateStudy(rates=[c.scan_rate for c in ordered])

    for curve in ordered:
        loop, _ = capacitance(curve)
        study.capacitances.append(loop.farads)

    span = max(study.rates) / min(study.rates)
    if span < MIN_RATE_SPAN:
        study.warnings.append(
            f"las velocidades sólo abarcan un factor {span:.1f}. Para separar "
            "un b de 0.8 de uno de 1.0 hace falta al menos una década: por "
            "debajo de eso el ajuste no distingue los dos casos que la "
            "pregunta quiere distinguir"
        )

    low = max(c.window[0] for c in ordered)
    high = min(c.window[1] for c in ordered)
    if high > low:
        targets = (
            list(b_potentials)
            if b_potentials is not None
            else list(np.linspace(low + 0.15 * (high - low), high - 0.15 * (high - low), 5))
        )
        for potential in targets:
            currents = [_current_at(c, potential) for c in ordered]
            usable = [
                (r, i) for r, i in zip(study.rates, currents)
                if i is not None and abs(i) > 0.0
            ]
            if len(usable) < 3:
                continue
            rates = np.array([r for r, _ in usable])
            values = np.array([abs(i) for _, i in usable])
            slope, intercept, error, r2 = _fit_line(np.log10(rates), np.log10(values))
            study.b_values.append(
                BValue(
                    potential=float(potential),
                    b=slope,
                    b_error=error,
                    a=10.0**intercept,
                    rates=tuple(rates),
                    r_squared=r2,
                )
            )
        study.capacitive_fraction = _dunn_split(ordered, low, high)

    _trasatti(study)
    _double_layer(study, ordered, ecsa_material, non_faradaic)
    return study


def _current_at(curve: Voltammogram, potential: float) -> Optional[float]:
    """Anodic current at one potential, interpolated on the forward sweep."""
    try:
        anodic, _ = curve.sweeps()
    except CurveError:
        return None
    order = np.argsort(anodic.potential)
    x, y = anodic.potential[order], anodic.current[order]
    if potential < x[0] or potential > x[-1]:
        return None
    return float(np.interp(potential, x, y))


def _dunn_split(
    curves: Sequence[Voltammogram], low: float, high: float
) -> dict[float, float]:
    """Dunn's separation ``i(V,ν) = k₁ν + k₂√ν`` at each scan rate.

    ``k₁ν`` is the surface-controlled (capacitive) part and ``k₂√ν`` the
    diffusion-controlled one. The split is done independently at a grid of
    potentials and integrated, which is the standard procedure — and it
    assumes the two mechanisms are the only ones and that neither has a
    scan-rate dependence of its own. Read the fractions as a comparison
    between samples measured the same way, not as absolutes.
    """
    potentials = np.linspace(low, high, 40)
    rates = np.array([c.scan_rate for c in curves])
    if rates.size < 3:
        return {}
    capacitive = np.zeros((len(curves), potentials.size))
    total = np.zeros_like(capacitive)
    for column, potential in enumerate(potentials):
        currents = [_current_at(c, potential) for c in curves]
        if any(value is None for value in currents):
            continue
        observed = np.array(currents, dtype=float)
        # i/sqrt(v) = k1*sqrt(v) + k2  -> a straight line in sqrt(v).
        root = np.sqrt(rates)
        slope, intercept, _, _ = _fit_line(root, observed / root)
        capacitive[:, column] = slope * rates
        total[:, column] = observed
    fractions: dict[float, float] = {}
    for index, curve in enumerate(curves):
        denominator = float(np.abs(total[index]).sum())
        if denominator <= 0.0:
            continue
        fractions[curve.scan_rate] = float(
            min(1.0, np.abs(capacitive[index]).sum() / denominator)
        )
    return fractions


def _trasatti(study: RateStudy) -> None:
    """Total and outer capacitance by Trasatti's double extrapolation.

    ``C(ν) = C_outer + k ν^(−1/2)`` extrapolated to ν → ∞ gives the
    capacitance of the surface the electrolyte reaches instantly;
    ``1/C(ν) = 1/C_total + k' ν^(1/2)`` extrapolated to ν → 0 gives the
    whole of it. The difference is the porosity that only slow scans see.

    The method assumes semi-infinite diffusion into the pores across the
    whole rate range, which is often violated at the slow end. Treat the
    two numbers as bounds rather than measurements.
    """
    if len(study.rates) < 4:
        study.warnings.append(
            "el análisis de Trasatti necesita al menos cuatro velocidades para "
            "que las dos extrapolaciones signifiquen algo"
        )
        return
    rates = np.array(study.rates, dtype=float)
    values = np.array(study.capacitances, dtype=float)
    if np.any(values <= 0.0):
        return
    outer_slope, outer_intercept, _, _ = _fit_line(1.0 / np.sqrt(rates), values)
    total_slope, total_intercept, _, _ = _fit_line(np.sqrt(rates), 1.0 / values)
    study.trasatti_outer = float(outer_intercept)
    study.trasatti_total = float(1.0 / total_intercept) if total_intercept > 0 else None
    if study.trasatti_total is not None and study.trasatti_outer is not None:
        if study.trasatti_outer > study.trasatti_total:
            study.warnings.append(
                "la extrapolación de Trasatti da una capacitancia externa MAYOR "
                "que la total, que es imposible. Significa que las hipótesis del "
                "método no se cumplen en tu rango de velocidades; no uses esos "
                "dos números"
            )


def _double_layer(
    study: RateStudy,
    curves: Sequence[Voltammogram],
    material: str,
    non_faradaic: bool,
) -> None:
    """C_dl from the slope of Δi/2 against scan rate, and the ECSA from it."""
    rates, currents = [], []
    for curve in curves:
        try:
            anodic, cathodic = curve.sweeps()
        except CurveError:
            continue
        centre = 0.5 * (curve.window[0] + curve.window[1])
        up = _current_at(curve, centre)
        order = np.argsort(cathodic.potential)
        down = float(np.interp(centre, cathodic.potential[order], cathodic.current[order]))
        if up is None:
            continue
        rates.append(curve.scan_rate)
        currents.append(0.5 * abs(up - down))
    if len(rates) < 3:
        return
    slope, _, _, r_squared = _fit_line(np.array(rates), np.array(currents))
    if slope <= 0.0:
        return
    study.cdl_farads = float(slope)

    data = load_echem_database()["ecsa_specific_capacitance_uF_cm2"]
    entry = data.get(material, data["por_defecto"])
    study.ecsa_cm2 = study.cdl_farads / (float(entry["value"]) * 1e-6)
    low, high = entry["range"]
    study.ecsa_range_cm2 = (
        study.cdl_farads / (float(high) * 1e-6),
        study.cdl_farads / (float(low) * 1e-6),
    )
    if r_squared < 0.98:
        study.warnings.append(
            f"la recta de C_dl tiene R² = {r_squared:.3f}. Si no es recta, la "
            "ventana no era puramente no faradaica y el C_dl lleva dentro "
            "corriente de reacción"
        )
    if not non_faradaic:
        study.warnings.append(
            "el C_dl y el ECSA se han calculado sin que se haya declarado que "
            "la ventana esté libre de corriente faradaica. Elige una región "
            "donde no pase nada (típicamente ±50 mV alrededor del potencial de "
            "circuito abierto) y vuelve a medir, o estos dos números incluyen "
            "reacción"
        )
    study.warnings.append(
        f"el ECSA se ha calculado con Cs = {entry['value']:g} µF/cm², pero la "
        f"literatura da entre {low:g} y {high:g} para el mismo tipo de "
        "material. Ese factor de tres pasa entero al área y de ahí a "
        "cualquier actividad que normalices por ella: es la magnitud peor "
        "determinada de toda la electrocatálisis. Compara ECSA entre TUS "
        "muestras, no contra las de un artículo"
    )


__all__ = [
    "MIN_RATE_SPAN",
    "PEAK_PROMINENCE",
    "VERTEX_TRIM",
    "BValue",
    "CVResult",
    "Capacitance",
    "RateStudy",
    "RedoxPeak",
    "analyse_cv",
    "analyse_rate_study",
    "capacitance",
    "find_redox_peaks",
    "integrate_charge",
]
