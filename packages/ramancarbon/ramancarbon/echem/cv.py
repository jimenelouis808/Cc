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
from scipy.optimize import lsq_linear as _lsq_linear
from scipy.stats import t as _student_t
from scipy.signal import find_peaks as _scipy_find_peaks
from scipy.signal import peak_widths as _peak_widths

from ..core.compat import trapezoid
from .curve import CurveError, Electrode, Voltammogram, load_echem_database

#: Minimum span in scan rate, as a ratio, for a b-value to mean anything.
MIN_RATE_SPAN = 8.0

#: Fraction of the potential window trimmed at each end before integrating.
#:
#: **Zero, and deliberately.** The received advice is to trim the vertices
#: because the potentiostat reverses in a finite time and the spike there
#: is instrument response. That is true of a slow acquisition; it is not
#: true of a modern averaged record, and trimming one costs real charge.
#: Measured on the user's files — 2000 points per cycle, ``<I>`` averaged
#: — there is no spike at all: the current is smooth through the vertex
#: and *falls* just after it, so a 2 % trim removed 7.5 % of the
#: capacitance at 5 mV/s and 2.9 % at 50 mV/s. The size of that loss
#: depends on how the sample's own current is distributed across the
#: window, which makes it a per-sample bias, not a correction.
#:
#: What replaces it is a measurement: :func:`vertex_transients` marks the
#: points that really are instrument response and nothing else is
#: discarded. Pass ``trim=`` explicitly to integrate a trimmed window
#: anyway.
VERTEX_TRIM = 0.0

#: A point is a switching transient if its |i| exceeds this multiple of
#: the median |i| of its own neighbourhood. A real vertex on these files
#: scores about 1.0; the charging spike left by stepping out of the
#: previous experiment scored 30.
TRANSIENT_FACTOR = 3.0

#: Size of that neighbourhood, and how close to a turning point a
#: transient has to be, as a fraction of the record. Away from a vertex a
#: large current is a redox peak.
TRANSIENT_REACH = 0.01

#: Minimum prominence of a redox peak, relative to the mean |current|.
PEAK_PROMINENCE = 0.08

#: Level of the test that calls a Dunn coefficient genuinely negative,
#: BEFORE dividing by how many coefficients are tested. A negative
#: coefficient is the two-term model failing, and the question is whether
#: it is negative by more than its own uncertainty — not by more than a
#: fixed fraction of its own scale, which on a coefficient that is truly
#: zero compares noise with noise.
NEGATIVE_LEVEL = 0.05

#: How far ``|q_cat|/q_an`` may stray from 1 before the two capacitance
#: conventions are reported as disagreeing for a physical reason.
BALANCE_TOLERANCE = 0.02

#: Two voltammograms count as being at the same scan rate when their
#: rates agree within this fraction. EC-Lab writes 49.99 mV/s for a
#: 50 mV/s setting, so an exact comparison finds no repeats at all.
SAME_RATE_TOLERANCE = 0.02

#: A repeat at the same scan rate that differs by more than this fraction
#: is the electrode changing, not scatter.
DRIFT_TOLERANCE = 0.05


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


#: A gap bigger than this many sampling steps is a join between
#: disjoint runs, not sweeping, so it takes no time and carries no
#: charge. The measured case is a thousand steps wide, so the exact
#: value is not a judgement call.
JOIN_STEPS = 10.0


def elapsed_time(curve: Voltammogram) -> tuple[np.ndarray, str]:
    """``(t, origen)``: the time axis to integrate against, in s.

    The instrument's own clock when the file carries one, and
    ``∫|dE|/ν`` only when it does not. That identity holds for a single
    uninterrupted sweep and for nothing else: any subset of a
    voltammogram — one branch, a trimmed window, a cycle whose counter
    started mid-sweep — is two or more disjoint runs, and the step
    between two runs is a potential the electrode never swept.

    Either axis is then **flattened at its joins**: a gap far larger than
    the sampling interval is set to zero duration, so the trapezoid
    across it contributes nothing. Measured on a real 50 mV/s cycle, the
    anodic branch held one 0.9998 V discontinuity and came back as
    39.99 s of sweeping instead of 20.00 — which is why the two
    capacitance conventions, which have to agree, were returning 5.6 and
    11.1 F/g for the same curve.
    """
    if curve.time is not None and curve.time.size == curve.n:
        axis = np.asarray(curve.time, dtype=float)
        axis = axis - float(axis[0])
        origin = "reloj del instrumento"
    else:
        axis = np.concatenate(
            ([0.0], np.cumsum(np.abs(np.diff(curve.potential))))
        ) / curve.scan_rate
        origin = "reconstruido de ∫|dE|/ν"

    gap = np.diff(axis)
    typical = float(np.median(gap)) if gap.size else 0.0
    if typical > 0.0:
        joins = gap > JOIN_STEPS * typical
        if joins.any():
            axis = np.concatenate(([0.0], np.cumsum(np.where(joins, 0.0, gap))))
            origin += f", {int(joins.sum())} empalme(s) descontado(s)"
    return axis, origin


def vertex_transients(curve: Voltammogram) -> tuple[np.ndarray, str]:
    """``(máscara, motivo)``: the points that are instrument response.

    A switching transient is a **discontinuity** in the current next to a
    turning point: the potentiostat reverses, the double layer is
    recharged in a few milliseconds, and the current falls back by a
    large factor within one or two samples. Both halves of that sentence
    are needed, and the first one is the one that is easy to get wrong.

    Testing only "is |i| large here" fails, because at a slow scan the
    sample's own current is largest exactly at the ends of the window —
    that is where the oxidation sets in. On the user's 5 mV/s file the
    current near the vertices reaches 6 times the cycle's median while
    rising perfectly smoothly, and a magnitude test discarded 62 real
    points and 8.7 % of the capacitance with them. What separates the two
    is the shape: a transient stands apart from the points on either side
    of it, a rising edge does not.

    So each candidate is compared with the median of its own
    neighbourhood, with its immediate neighbours left out so that a
    one-to-three-point spike cannot defend itself. A smooth edge scores
    about 1 and is kept; the charging spike left by stepping out of the
    previous experiment scored 30 and is dropped.

    The mask is ``True`` where a point is to be discarded. On the user's
    reporting cycles it is empty, and the reason says so.
    """
    size = curve.n
    magnitude = np.abs(curve.current)
    if float(np.median(magnitude)) <= 0.0:
        return np.zeros(size, dtype=bool), "corriente nula"

    direction = np.sign(np.diff(curve.potential))
    direction[direction == 0] = 1
    turns = np.flatnonzero(np.diff(direction) != 0) + 1
    edges = np.concatenate(([0], turns, [size - 1]))
    reach = max(3, int(round(TRANSIENT_REACH * size)))

    candidates: set[int] = set()
    for edge in edges:
        candidates.update(range(max(0, edge - reach), min(size, edge + reach + 1)))

    mask = np.zeros(size, dtype=bool)
    worst = 0.0
    for index in sorted(candidates):
        before = magnitude[max(0, index - reach):max(0, index - 1)]
        after = magnitude[index + 2:index + 2 + reach]
        neighbourhood = np.concatenate([before, after])
        if neighbourhood.size < 3:
            continue
        reference = float(np.median(neighbourhood))
        if reference <= 0.0:
            continue
        ratio = float(magnitude[index]) / reference
        worst = max(worst, ratio)
        if ratio > TRANSIENT_FACTOR:
            mask[index] = True

    if not mask.any():
        return mask, (
            f"sin transitorio de conmutación: en los vértices el |i| más "
            f"destacado es {worst:.2f} veces la mediana de su propio "
            f"entorno, por debajo del umbral de {TRANSIENT_FACTOR:g}, así "
            f"que no se recorta nada"
        )
    return mask, (
        f"{int(mask.sum())} punto(s) descartados junto a un vértice por "
        f"destacar más de {TRANSIENT_FACTOR:g} veces sobre su entorno "
        f"(máximo {worst:.1f} veces): respuesta del instrumento, no de la "
        f"muestra"
    )


def integrate_charge(
    curve: Voltammogram, trim: float = VERTEX_TRIM, whole_file: bool = False
) -> tuple[float, float, float]:
    """``(anodic charge, absolute loop charge, integrated span)`` in C and V.

    ``q = ∫ i dt``, against the clock of :func:`elapsed_time`. Points that
    :func:`vertex_transients` identifies as instrument response are
    dropped; ``trim`` additionally removes that fraction of the window at
    each vertex and defaults to nothing — :data:`VERTEX_TRIM` says why.

    **One cycle, not the file.** A multi-cycle record is integrated over
    its reporting cycle (:meth:`Voltammogram.reporting_cycle`). Before
    this, the charge was accumulated over every cycle in the array and
    divided by ONE window, so the capacitance came out multiplied by the
    number of cycles — measured on a real four-cycle file at 50 mV/s,
    22.77 F/g against the 5.65–5.71 F/g its individual cycles give, a
    factor of 4.03. Nothing in the result said how many cycles the file
    held, so the number looked like any other number.

    ``whole_file=True`` restores the old behaviour, for the caller who
    really does want the charge passed over the whole record. It is not
    a capacitance.
    """
    if not whole_file:
        curve, _ = curve.reporting_cycle()
    keep = ~vertex_transients(curve)[0]
    if trim > 0.0:
        low, high = curve.window
        margin = trim * (high - low)
        keep &= (curve.potential >= low + margin) & (curve.potential <= high - margin)
    if keep.sum() < 10:
        keep = np.ones(curve.n, dtype=bool)
    curve = curve._subset(keep, "integrado")

    time, _ = elapsed_time(curve)
    current = curve.current
    anodic = np.where(current > 0.0, current, 0.0)
    # The span actually integrated, which is what the charge must be
    # divided by. Trimming the vertices and then dividing by the FULL
    # window makes every capacitance 2 x trim too small — 4 % with the
    # old default, a systematic error that looks like a real difference
    # between samples measured with different vertex settings.
    span = float(curve.potential.max() - curve.potential.min())
    return (
        float(trapezoid(anodic, time)),
        float(trapezoid(np.abs(current), time)),
        span,
    )


def coulombic_efficiency(curve: Voltammogram) -> Optional[float]:
    """``Q_red / Q_ox`` over the reporting cycle, or ``None``.

    Total reduction charge against total oxidation charge, by the sign of
    the current and not by which sweep it was on — a current that stays
    anodic past the vertex is oxidation, whichever direction the potential
    is going.

    This is the first number to look at before quoting a capacitance,
    because a capacitor returns what it takes: 1.00 is the ideal, and
    whatever is missing was spent on something irreversible. On the user's
    rate series it runs 0.48 at 5 mV/s, 0.69 at 10, 0.82 at 20 and
    0.84–0.90 at 50 — at the slow end **half the charge does not come
    back**, so half of what the integral calls capacitance is an
    irreversible reaction and the F/g figure describes that reaction as
    much as the double layer.

    It also explains, without being equal to it, the gap between the two
    conventions of :func:`capacitance`: both are computed from an
    unbalanced loop, and neither convention can repair it.
    """
    curve, _ = curve.reporting_cycle()
    anodic, absolute, _ = integrate_charge(curve)
    cathodic = absolute - anodic
    if anodic <= 0.0:
        return None
    return cathodic / anodic


def capacitance(
    curve: Voltammogram, window: Optional[tuple[float, float]] = None
) -> tuple[Capacitance, Capacitance]:
    """Capacitance from a voltammogram, in both usual conventions.

    Returns ``(loop, sweep)``. Numerically the two agree; they are computed
    separately so that a mismatch — which happens when the loop is not
    closed, or when one branch has a faradaic contribution the other lacks
    — is visible instead of averaged away.

    Computed over ONE cycle, the last complete one, and which one is on
    the result's ``note``. See :func:`integrate_charge`.
    """
    curve, cycle_note = curve.reporting_cycle()
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

    # The two conventions agree only on a charge-balanced loop. When they
    # do not, the gap IS the imbalance, and saying so is the whole point
    # of computing both: it is a property of the sample, not a numerical
    # disagreement to be split.
    efficiency = coulombic_efficiency(curve)
    if efficiency is not None and abs(1.0 - efficiency) > BALANCE_TOLERANCE:
        gap = 100.0 * (sweep_value / loop_value - 1.0) if loop_value else 0.0
        note = (
            f"eficiencia coulómbica Q_red/Q_ox = {efficiency:.3f}: el lazo NO "
            f"está equilibrado. Un condensador devuelve lo que toma, así que "
            f"el {100.0 * (1.0 - efficiency):.0f} % que falta se gastó en algo "
            f"irreversible y entra en esta integral igual que la doble capa: "
            f"la cifra en F/g describe esa reacción tanto como el "
            f"almacenamiento. Es también el origen del {gap:+.1f} % que "
            f"separa los dos convenios, y ningún convenio lo arregla. El del "
            f"lazo cerrado, que promedia las dos ramas, es el conservador"
        )
        cycle_note = f"{cycle_note}. {note}" if cycle_note else note

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
                note=cycle_note,
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
    drift: Optional["DriftCheck"] = None
    """The repeat at an already-measured rate, when the series has one."""
    leakage: Optional["LeakageFit"] = None
    """The constant parasitic current the series implies, when three or
    more rates make it fittable."""
    current_laws: list["CurrentLaw"] = field(default_factory=list)
    """Which law for ``i(ν)`` actually wins at each potential. Read this
    before the b-values above: they assume one of the three."""
    full_model: list["FullCurrentModel"] = field(default_factory=list)
    """``k_cap·ν + k_diff·√ν + i_par`` at each potential, when there are
    four rates or more. The two-term laws make you pick two of the
    three; this one does not, and says which terms the series resolves."""
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
            if self.drift is not None and self.drift.degraded:
                lines.append(
                    f"  ⚠ esa retención incluye la degradación: "
                    f"{self.drift.fade_percent:+.1f} % medidos en la "
                    f"repetición a {1e3 * self.drift.rate:g} mV/s"
                )
        if self.current_laws:
            lines.append("")
            lines.append(
                "¿Qué ley describe i(ν)?  (tres modelos de dos parámetros, "
                "residuos comparables)"
            )
            lines.extend("  " + str(law) for law in self.current_laws)
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
        if self.full_model:
            lines.append("")
            lines.append(
                "Las tres contribuciones a la vez "
                "(i = k_cap·ν + k_diff·√ν + i_par):"
            )
            lines.extend("  " + str(model) for model in self.full_model)
        if self.leakage is not None:
            lines.append("")
            lines.append(str(self.leakage))
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


@dataclass
class DriftCheck:
    """What a repeat at an already-measured scan rate says.

    A rate series is run in order, and the order is almost always from
    slow to fast or fast to slow — which means elapsed time and scan rate
    move together. If the electrode changes during the series, the change
    enters the rate dependence and comes back out as kinetics: a b-value,
    a capacitive fraction, a retention figure. Nothing in those numbers
    says it happened.

    One repeat at a rate already measured separates the two, and it is the
    only thing in the series that can. Measured on the user's files, the
    two 50 mV/s records — the first experiment of the session and the
    last — give 7.17 and 5.81 F/g: the electrode lost 19 % of its
    capacitance over the 22 cycles between them, so a good part of the
    apparent rate dependence of that series is fade.
    """

    rate: float
    """V/s, the repeated rate."""
    first: float
    last: float
    """Capacitance in F at the first and last measurement of that rate."""
    names: tuple[str, str] = ("", "")

    @property
    def fade_percent(self) -> float:
        """Positive when the electrode lost capacitance."""
        if self.first <= 0.0:
            return 0.0
        return 100.0 * (1.0 - self.last / self.first)

    @property
    def degraded(self) -> bool:
        return abs(self.fade_percent) > 100.0 * DRIFT_TOLERANCE

    def __str__(self) -> str:
        head = (
            f"repetición a {1e3 * self.rate:g} mV/s: "
            f"{1e3 * self.first:.4g} → {1e3 * self.last:.4g} mF "
            f"({self.fade_percent:+.1f} %)"
        )
        if not self.degraded:
            return (
                head + f" — dentro del {100 * DRIFT_TOLERANCE:g} %, el "
                "electrodo no cambió durante la serie y la dependencia con "
                "ν es cinética"
            )
        return (
            head + " — el electrodo CAMBIÓ durante la serie. Las velocidades "
            "se midieron en orden, así que el tiempo y ν van juntos y esta "
            "pérdida entra en la dependencia con ν: el b, el reparto de Dunn "
            "y la retención la llevan dentro sin decirlo. Lo que la serie "
            "mide es cinética MÁS degradación, y sólo esta repetición las "
            "separa"
        )


def within_file_fade(curve: Voltammogram) -> Optional[tuple[float, str]]:
    """``(fade % per cycle, texto)`` across the complete cycles of one file.

    Cheaper than a repeat and available on every multi-cycle record: if
    the capacitance of a file's own cycles falls monotonically, the
    reporting cycle is a point on a slope and not a steady state. On the
    user's 5 mV/s file it falls 12.20 → 11.33 F/g over four cycles, 2.4 %
    per cycle, so the quoted value depends on which cycle is quoted.
    """
    complete = curve.complete_cycles()
    if len(complete) < 3:
        return None
    values = []
    for cycle in complete:
        try:
            values.append(capacitance(cycle)[0].farads)
        except CurveError:
            return None
    array = np.array(values, dtype=float)
    if array[0] <= 0.0:
        return None
    per_cycle = 100.0 * (1.0 - (array[-1] / array[0]) ** (1.0 / (array.size - 1)))
    monotone = bool(np.all(np.diff(array) < 0.0) or np.all(np.diff(array) > 0.0))
    text = (
        f"los {array.size} ciclos completos del archivo dan "
        + " → ".join(f"{1e3 * v:.4g}" for v in array)
        + f" mF, {per_cycle:+.2f} % por ciclo"
    )
    if monotone and abs(per_cycle) > 0.5:
        text += (
            ". La caída es MONÓTONA: el ciclo que se informa está sobre una "
            "pendiente, no en estado estacionario, y el valor citado depende "
            "de cuál se cite. Para una cifra publicable hace falta ciclar "
            "hasta que se estabilice"
        )
    return per_cycle, text


def reversible_capacitance(curve: Voltammogram) -> Capacitance:
    """Capacitance from the REDUCTION charge alone, ``Q_red / ΔV``.

    The usual conventions integrate the oxidation charge, or both halves
    averaged. Either is the wrong thing to integrate when the electrode
    carries a parasitic *anodic* current — corrosion, carbon oxidation,
    electrolyte oxidation at the positive limit — because that current
    enters the integral and grows with how long the cycle takes, so the
    slow scans come out larger and the result looks like a capacitance
    with a rate dependence.

    Reduction is the half that parasitic current does not contribute to,
    and what comes back out is by definition what was stored. On the
    user's files this is the difference between a plausible number and a
    misleading one: the loop convention gives 11.33 F/g at 5 mV/s falling
    to 5.81 at 50, a 51 % retention that looks like a transport
    limitation, while the reduction charge gives 7.39 → 5.52, with the
    whole of the remaining difference accounted for by the 16 % the
    electrode itself lost between the two measurements.

    Use it together with :func:`coulombic_efficiency`: when that is 1.00
    this agrees with the other two conventions and adds nothing, and when
    it is not, this is the one to quote.
    """
    curve, cycle_note = curve.reporting_cycle()
    anodic, absolute, span = integrate_charge(curve)
    if span <= 0.0:
        raise CurveError("la ventana de potencial es nula")
    reduction = absolute - anodic
    value = reduction / span
    electrode = curve.electrode
    efficiency = coulombic_efficiency(curve)
    note = cycle_note
    if efficiency is not None:
        extra = (
            f"Q_red = {reduction:.4g} C sobre ΔV = {span:.4g} V, con "
            f"Q_red/Q_ox = {efficiency:.3f}"
        )
        note = f"{note}. {extra}" if note else extra
    return Capacitance(
        farads=value,
        convention="carga de reducción ÷ ΔV",
        specific_f_per_g=electrode.specific(value, "mass"),
        specific_f_per_cm2=electrode.specific(value, "area"),
        specific_f_per_cm3=electrode.specific(value, "volume"),
        note=note,
    )


#: Models for ``i(ν)`` at a fixed potential, each with two parameters so
#: that their residuals can be compared without penalising any of them.
#:
#: The first two are the ones everybody fits. The third is the one almost
#: nobody fits, and on the user's files it fits exactly as well as the
#: other two — which is the whole reason it is here.
CURRENT_LAWS: tuple[tuple[str, str], ...] = (
    ("potencia", "i = a·ν^b"),
    ("dunn", "i = k₁·ν + k₂·√ν"),
    ("fuga", "i = k_cap·ν + i_par"),
)


@dataclass
class CurrentLaw:
    """Which of :data:`CURRENT_LAWS` describes ``i(ν)`` at one potential.

    The b-value and the Dunn separation are both answers to the question
    "how does the current scale with scan rate", and both assume the
    answer is a sum of a term in ν and a term in √ν. A constant current —
    corrosion, carbon oxidation, electrolyte breakdown near the positive
    limit, anything that does not care how fast the potential is moving —
    is neither. Over a factor of ten in ν, though, a constant and a
    ``k₂√ν`` are nearly the same function, so fitting the constant
    explicitly is the only way to find out whether it is there.

    **The usual answer is that the series cannot tell.** On the user's
    four rates all three laws come back at R² between 0.989 and 0.998 at
    every potential in the window, and the differences between them are
    far smaller than what four points and two parameters can resolve.
    That is a result and not a failure: a decade of scan rates is the
    minimum for a b-value to mean anything, and it is still not enough to
    choose the LAW the b-value belongs to.

    So the tie is broken from outside this fit, and on these files it
    breaks against diffusion. The parasitic current is not inferred here,
    it is measured independently — :func:`coulombic_efficiency` is 0.48 at
    5 mV/s, and :class:`LeakageFit` gives ⟨i_par⟩ = 0.41 µA at R² = 0.996
    with an intercept of −9 µC against charges of 10–150 µC. The leak law
    then returns, at each potential, a capacitance that barely varies
    across the window (128 µF at −0.40 V, 95 µF at −0.20 V) and a
    constant that climbs monotonically towards the positive limit
    (0.30 → 2.30 µA): the right magnitude, and the right shape for an
    irreversible oxidation. Diffusion control is *permitted* by the
    scan-rate fit and nothing else in the data asks for it, while the
    leak is measured three different ways.

    Which matters because b falls from 0.96 at −0.30 V to 0.46 at
    +0.30 V across that same window, and 0.46 is read off as
    "diffusion-controlled, battery-like, report in mAh/g". The data do
    not support that reading, and they do support the leak.
    """

    potential: float
    residuals: dict[str, float] = field(default_factory=dict)
    """Sum of squared residuals in A², per law key. Comparable: every law
    here has two parameters."""
    b: float = 0.0
    """Exponent of the power law, for continuity with :class:`BValue`."""
    capacitance_f: float = 0.0
    """``k_cap`` of the leak law, in F."""
    leakage_a: float = 0.0
    """``i_par`` of the leak law, in A."""
    r_squared: dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> str:
        """Key of the law with the smallest residual."""
        if not self.residuals:
            return ""
        return min(self.residuals, key=lambda key: self.residuals[key])

    @property
    def decisive(self) -> bool:
        """Whether the winner beats the runner-up by more than a third.

        Four points and two parameters leave two degrees of freedom, so a
        small difference in residual is not a result.
        """
        ordered = sorted(self.residuals.values())
        if len(ordered) < 2 or ordered[1] <= 0.0:
            return False
        return ordered[0] < 0.67 * ordered[1]

    def __str__(self) -> str:
        names = dict(CURRENT_LAWS)
        text = (
            f"E = {self.potential:+.3f} V:  b = {self.b:.3f}   "
            + "  ".join(
                f"R²({key}) = {self.r_squared.get(key, float('nan')):.5f}"
                for key, _ in CURRENT_LAWS
            )
        )
        if self.decisive:
            text += f"  →  gana «{names.get(self.best, self.best)}»"
        else:
            text += "  →  EMPATE: la serie no distingue las tres leyes"
        text += (
            f"; la de fuga pide C = {1e6 * self.capacitance_f:.0f} µF más "
            f"{1e6 * self.leakage_a:+.2f} µA constantes"
        )
        return text


def compare_current_laws(
    curves: Sequence[Voltammogram],
    potentials: Optional[Sequence[float]] = None,
    sweep: str = "media",
) -> list[CurrentLaw]:
    """Fit all of :data:`CURRENT_LAWS` at each potential and rank them.

    Run this before quoting a b-value or a Dunn percentage. Both of those
    presuppose that the current is a sum of a surface term and a
    diffusion term; this says whether that presupposition survives
    contact with the data, and on an electrode with a parasitic current
    it does not.
    """
    usable = [c for c in curves if c.scan_rate > 0.0]
    if len(usable) < 4:
        return []
    ordered = sorted(usable, key=lambda c: c.scan_rate)
    rates = np.array([c.scan_rate for c in ordered], dtype=float)

    if potentials is None:
        low = max(c.window[0] for c in ordered)
        high = min(c.window[1] for c in ordered)
        if high <= low:
            return []
        potentials = list(np.linspace(low + 0.1 * (high - low),
                                      high - 0.1 * (high - low), 9))

    out: list[CurrentLaw] = []
    for potential in potentials:
        sampled = [_current_at(c, float(potential), sweep) for c in ordered]
        if any(value is None for value in sampled):
            continue
        current = np.array([abs(value) for value in sampled], dtype=float)
        if np.any(current <= 0.0):
            continue
        total = float(((current - current.mean()) ** 2).sum())
        if total <= 0.0:
            continue

        law = CurrentLaw(potential=float(potential))
        designs = {
            "dunn": np.vstack([rates, np.sqrt(rates)]).T,
            "fuga": np.vstack([rates, np.ones_like(rates)]).T,
        }
        for key, design in designs.items():
            coefficients, *_ = np.linalg.lstsq(design, current, rcond=None)
            residual = float(((current - design @ coefficients) ** 2).sum())
            law.residuals[key] = residual
            law.r_squared[key] = 1.0 - residual / total
            if key == "fuga":
                law.capacitance_f = float(coefficients[0])
                law.leakage_a = float(coefficients[1])

        # The power law is fitted in logarithms, as it always is, but its
        # residual is measured on the current itself so that it can be
        # compared with the other two.
        slope, intercept, _, _ = _fit_line(np.log10(rates), np.log10(current))
        law.b = slope
        predicted = 10.0**intercept * rates**slope
        residual = float(((current - predicted) ** 2).sum())
        law.residuals["potencia"] = residual
        law.r_squared["potencia"] = 1.0 - residual / total
        out.append(law)
    return out


@dataclass
class FullCurrentModel:
    """``i = k_cap·ν + k_diff·√ν + i_par`` at one potential: all three.

    :func:`compare_current_laws` asks which of three two-parameter laws
    describes the current, and over a single decade of scan rate the
    answer is usually "cannot tell". That comparison is a screening, and
    it is the wrong question once the series is wide enough: a real
    electrode can perfectly well have a double layer AND a diffusive
    process AND a parasitic current at the same time, and the two-term
    laws make you pick two of the three.

    This fits all three. It needs a wider series to be worth anything —
    four rates and three parameters leave one degree of freedom, which
    is not a fit — and what it returns is as much about which
    coefficients are DETERMINED as about their values. A term whose
    standard error is as large as itself is a term the series cannot
    see, and saying so is the result.

    Measured on the user's 5–100 mV/s series, where the two-term
    screening says the Dunn form beats a constant at 7 of 9 potentials:
    the three-term fit keeps all three terms, with the parasitic current
    landing on the 0.40 µA the charge balance measured independently.
    On the earlier 5–50 mV/s series, a factor of ten, the same fit
    returns errors larger than the coefficients.
    """

    potential: float
    capacitance_f: float
    """``k_cap`` in F: the surface-controlled term."""
    diffusive: float
    """``k_diff`` in A/(V/s)^½."""
    leakage_a: float
    """``i_par`` in A: the term that does not care about ν."""
    errors: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Standard errors of the three, in their own units."""
    r_squared: float = 0.0
    degrees_of_freedom: int = 0
    at_bound: tuple[bool, bool, bool] = (False, False, False)
    """Which coefficients the non-negativity constraint is holding at
    zero. One of these is the fit saying that term is not there."""

    def determined(self) -> dict[str, bool]:
        """Which terms the series actually resolves.

        A coefficient counts as determined when it is not sitting on its
        bound and is at least twice its own standard error in magnitude.
        Two sigma is a low bar on purpose: this is the threshold of
        "this number means nothing", not of "this number is good". The
        magnitude, not the value, because ``i_par`` is legitimately
        negative wherever the parasitic reaction is a reduction.
        """
        values = (self.capacitance_f, self.diffusive, self.leakage_a)
        names = ("k_cap", "k_diff", "i_par")
        return {
            name: bool(not bound and error > 0.0 and abs(value) > 2.0 * error)
            for name, value, error, bound
            in zip(names, values, self.errors, self.at_bound)
        }

    def __str__(self) -> str:
        found = self.determined()
        parts = []
        for name, value, error, scale in (
            ("k_cap", self.capacitance_f, self.errors[0], 1e6),
            ("k_diff", self.diffusive, self.errors[1], 1e6),
            ("i_par", self.leakage_a, self.errors[2], 1e6),
        ):
            mark = "" if found[name] else "  (no determinado)"
            parts.append(f"{name} = {scale * value:.3g} ± {scale * error:.2g}{mark}")
        return (f"E = {self.potential:+.3f} V  (R² = {self.r_squared:.5f}, "
                f"{self.degrees_of_freedom} gl):  " + ";  ".join(parts))


def fit_full_current(
    curves: Sequence[Voltammogram],
    potentials: Optional[Sequence[float]] = None,
    sweep: str = "media",
) -> list[FullCurrentModel]:
    """Fit :class:`FullCurrentModel` at each potential.

    Returns ``[]`` with fewer than four scan rates: three parameters and
    four points leave one degree of freedom, and a standard error from
    one degree of freedom is not a standard error. Five rates over a
    factor of twenty is the smallest series this has been seen to work
    on.
    """
    usable = [c for c in curves if c.scan_rate > 0.0]
    ordered = sorted(unique_rates(usable), key=lambda c: c.scan_rate)
    if len(ordered) < 4:
        return []
    rates = np.array([c.scan_rate for c in ordered], dtype=float)
    design = np.vstack([rates, np.sqrt(rates), np.ones_like(rates)]).T
    dof = len(ordered) - 3

    if potentials is None:
        low = max(c.window[0] for c in ordered)
        high = min(c.window[1] for c in ordered)
        if high <= low:
            return []
        potentials = list(np.linspace(low + 0.1 * (high - low),
                                      high - 0.1 * (high - low), 9))

    out: list[FullCurrentModel] = []
    for potential in potentials:
        sampled = [_current_at(c, float(potential), sweep) for c in ordered]
        if any(value is None for value in sampled):
            continue
        signed = np.array(sampled, dtype=float)
        if not np.any(signed):
            continue
        # The branch's own direction, applied to the DATA, with the
        # coefficients turned back at the end. Taking |i| instead looks
        # equivalent and is not: where the parasitic current is large
        # and opposite to the sweep, |i| folds the sum about zero and
        # the three-term model stops holding. On a synthetic series with
        # a −0.8 µA constant it came back as −8.0.
        direction = float(np.sign(np.median(signed))) or 1.0
        current = direction * signed
        # k_cap and k_diff are magnitudes of a capacitive and a
        # diffusive response and cannot be negative. **i_par can.** A
        # parasitic current has whatever sign its reaction has, and on
        # this window it changes: at the negative limit the electrode
        # sits in the oxygen-reduction region and the constant is
        # CATHODIC, at the positive limit it is evolving oxygen and the
        # constant is anodic. Measured on the user's 5–100 mV/s series,
        # it runs −1.5 µA at −0.40 V to +1.2 µA at +0.30 V and crosses
        # zero near +0.05 V. Forcing it positive made the fit put 11 µF
        # of capacitance where there are 90 and called the rest a leak.
        solution = _lsq_linear(design, current,
                               bounds=([0.0, 0.0, -np.inf],
                                       [np.inf, np.inf, np.inf]))
        coefficients = solution.x
        residual = current - design @ coefficients
        spread = float(((current - current.mean()) ** 2).sum())
        r_squared = (1.0 - float((residual**2).sum()) / spread
                     if spread > 0 else 1.0)
        # Errors from the UNCONSTRAINED covariance. Where the constraint
        # binds the error is not meaningful anyway, and `at_bound` says
        # where that is, so an approximate error beside a flag is more
        # honest than an exact one nobody can interpret.
        variance = float((residual**2).sum()) / dof if dof > 0 else 0.0
        try:
            covariance = variance * np.linalg.inv(design.T @ design)
            errors = tuple(float(np.sqrt(max(covariance[i, i], 0.0)))
                           for i in range(3))
        except np.linalg.LinAlgError:
            errors = (float("inf"),) * 3
        out.append(FullCurrentModel(
            potential=float(potential),
            capacitance_f=direction * float(coefficients[0]),
            diffusive=direction * float(coefficients[1]),
            leakage_a=direction * float(coefficients[2]),
            errors=errors,
            r_squared=r_squared,
            degrees_of_freedom=dof,
            # Only the two that HAVE a bound can sit on one.
            at_bound=(bool(np.isclose(coefficients[0], 0.0)),
                      bool(np.isclose(coefficients[1], 0.0)),
                      False),
        ))
    return out


@dataclass
class LeakageFit:
    """``ΔQ = Q_ox − Q_red = ⟨i_par⟩ · t`` across a series of scan rates.

    The net charge of a cycle — oxidation minus reduction — is
    ``∫ i dt`` over the loop, and that is *by definition* the charge the
    electrode passed and did not give back. A stored charge contributes
    nothing to it, whatever the storage mechanism; only an irreversible
    process does. So the slope of the net charge against the cycle's
    duration is the parasitic current, **averaged over the cycle**: if
    the reaction only runs near the positive limit, this is its average
    and not its instantaneous value, which is the right quantity to quote
    anyway, because it is what the electrode loses per second of
    operation.

    Fitting it is worth the trouble because the leak and the storage are
    not separable from any single voltammogram, and because a one-sided
    leak is what makes a capacitance look rate-dependent when it is not:
    it adds to ``Q_ox`` without adding to ``Q_red``, and the longer the
    cycle, the more it adds. On the user's four rates the fit gives
    ⟨i_par⟩ = 0.41 µA at R² = 0.998 with an intercept of −5 µC against
    charges of 10–150 µC, so the model holds; and 65 % of the oxidation
    charge at 5 mV/s is that leak rather than storage.
    """

    leakage_a: float
    """⟨i_par⟩ in A, the cycle-averaged irreversible current."""
    offset_c: float
    """Intercept of the fit in C. The model says zero, so this is the
    check on it rather than a result."""
    r_squared: float
    reduction_charges: tuple[float, ...] = ()
    """``Q_red`` per rate, the charge that did come back out."""
    oxidation_charges: tuple[float, ...] = ()
    rates: tuple[float, ...] = ()
    durations: tuple[float, ...] = ()
    span_v: float = 0.0
    electrode: Optional["Electrode"] = None

    @property
    def reversible_charge_c(self) -> float:
        """Mean ``Q_red``, which is what the electrode actually stores.

        Its spread across the series is not fitting error: it is the
        electrode changing while the series was measured. Read it with
        :class:`DriftCheck`.
        """
        if not self.reduction_charges:
            return 0.0
        return float(np.mean(self.reduction_charges))

    @property
    def capacitance_farads(self) -> Optional[float]:
        """The leakage-free capacitance, ``⟨Q_red⟩ / ΔV``."""
        if self.span_v <= 0.0:
            return None
        return self.reversible_charge_c / self.span_v

    @property
    def shares(self) -> dict[float, float]:
        """Scan rate → fraction of ``Q_ox`` that was the leak."""
        out: dict[float, float] = {}
        for rate, duration, oxidation in zip(
            self.rates, self.durations, self.oxidation_charges
        ):
            if oxidation > 0.0:
                out[rate] = self.leakage_a * duration / oxidation
        return out

    def __str__(self) -> str:
        lines = [
            "Corriente parásita (ΔQ = Q_ox − Q_red = ⟨i_par⟩·t):",
            f"  ⟨i_par⟩ = {1e6 * self.leakage_a:.3g} µA",
            f"  ordenada = {1e6 * self.offset_c:.3g} µC "
            f"(el modelo dice cero)",
            f"  R²      = {self.r_squared:.4f}",
        ]
        value = self.capacitance_farads
        if value is not None:
            spread = (
                float(np.std(self.reduction_charges)) / self.span_v
                if len(self.reduction_charges) > 1 else 0.0
            )
            text = f"  C sin fuga = {1e3 * value:.4g} ± {1e3 * spread:.3g} mF"
            if self.electrode is not None:
                specific = self.electrode.specific(value, "mass")
                error = self.electrode.specific(spread, "mass")
                if specific is not None:
                    text += f" = {specific:.3g}"
                    if error:
                        text += f" ± {error:.2g}"
                    text += " F/g"
            lines.append(text)
            lines.append(
                "  (de ⟨Q_red⟩; la dispersión es el electrodo cambiando "
                "durante la serie, no error de ajuste)"
            )
        for rate, share in sorted(self.shares.items()):
            lines.append(
                f"  {1e3 * rate:7.1f} mV/s   la fuga es el {100 * share:4.0f} % "
                f"de Q_ox"
            )
        if self.r_squared < 0.95:
            lines.append(
                "  ⚠ R² < 0.95: la corriente parásita no es constante en el "
                "tiempo, así que este reparto no la describe. Mide una "
                "cronoamperometría al potencial del límite positivo para "
                "verla directamente"
            )
        return "\n".join(lines)


def leakage_from_rates(curves: Sequence[Voltammogram]) -> Optional[LeakageFit]:
    """Fit :class:`LeakageFit` over a rate series.

    ``None`` with fewer than three usable rates: two points fit a line
    exactly and say nothing about whether it is a line, and the intercept
    is the whole check.
    """
    durations: list[float] = []
    net: list[float] = []
    oxidation_charges: list[float] = []
    reduction_charges: list[float] = []
    rates: list[float] = []
    spans: list[float] = []
    electrode: Optional[Electrode] = None
    for curve in curves:
        cycle, _ = curve.reporting_cycle()
        time, _ = elapsed_time(cycle)
        if time.size < 2 or time[-1] <= 0.0:
            continue
        oxidation, absolute, span = integrate_charge(cycle)
        if span <= 0.0:
            continue
        reduction = absolute - oxidation
        durations.append(float(time[-1]))
        net.append(float(oxidation - reduction))
        oxidation_charges.append(float(oxidation))
        reduction_charges.append(reduction)
        rates.append(float(curve.scan_rate))
        spans.append(span)
        electrode = curve.electrode
    if len(durations) < 3:
        return None
    slope, intercept, _, r_squared = _fit_line(
        np.array(durations), np.array(net)
    )
    return LeakageFit(
        leakage_a=slope,
        offset_c=intercept,
        r_squared=r_squared,
        reduction_charges=tuple(reduction_charges),
        oxidation_charges=tuple(oxidation_charges),
        rates=tuple(rates),
        durations=tuple(durations),
        span_v=float(np.mean(spans)),
        electrode=electrode,
    )


def without_parasitic(
    curves: Sequence[Voltammogram], current: float
) -> tuple[list[Voltammogram], str]:
    """Subtract a constant anodic current from every curve.

    ``current`` in A, **measured independently of ``i(ν)``**. That
    requirement is the whole point and it is easy to miss: the b-value
    and the Dunn separation are fits of ``i(ν)``, so correcting them with
    a parasitic current that was itself fitted to ``i(ν)`` assumes the
    answer. Subtract ``k_cap·ν + i_par``'s own ``i_par`` and Dunn
    afterwards returns 100 % capacitive by construction, having been
    told so.

    Two measurements qualify. :class:`LeakageFit` gets the current from
    the CHARGE balance, ``ΔQ`` against cycle duration, which never looks
    at how the current scales with ν — that is the one used below. A
    chronoamperometry held at the positive limit measures it even more
    directly, and is what to do when the question matters.

    What comes back is a copy of each curve with the constant removed,
    and a sentence saying what was done, for the figure caption. Run the
    rate study again on it: if b climbs to 1 and the Dunn fractions go
    with it, the departure was the leak; if it does not, something else
    is going on and the measurement can say so.
    """
    corrected = []
    for curve in curves:
        clone = Voltammogram(
            potential=curve.potential,
            current=curve.current - float(current),
            scan_rate=curve.scan_rate,
            electrode=curve.electrode,
            cycle=curve.cycle,
            time=curve.time,
            name=f"{curve.name} [sin fuga]",
            metadata=dict(curve.metadata),
        )
        clone.metadata["corriente_parasita_restada_a"] = float(current)
        corrected.append(clone)
    return corrected, (
        f"restada una corriente anódica constante de "
        f"{1e6 * current:.2f} µA, medida del balance de carga (ΔQ contra "
        f"la duración del ciclo) y por tanto independiente de cómo escala "
        f"i con ν"
    )


def unique_rates(curves: Sequence[Voltammogram]) -> list[Voltammogram]:
    """One curve per scan rate, keeping the LAST of each.

    The input order is the measurement order, so the last of a repeated
    rate is the one measured inside the series rather than before it.
    Keeping both would put the same rate twice into a fit whose whole job
    is to describe how things change WITH the rate: the repeat adds no
    information to the line it is collinear with, and it quietly doubles
    that rate's weight. Its information is in :func:`drift_check`.
    """
    kept: list[Voltammogram] = []
    for curve in curves:
        for index, existing in enumerate(kept):
            if existing.scan_rate > 0.0 and abs(
                curve.scan_rate / existing.scan_rate - 1.0
            ) <= SAME_RATE_TOLERANCE:
                kept[index] = curve
                break
        else:
            kept.append(curve)
    return kept


def drift_check(curves: Sequence[Voltammogram]) -> Optional[DriftCheck]:
    """The repeat in a rate series, if there is one.

    ``None`` when every rate appears once, which is the usual case and the
    reason the series cannot be checked.
    """
    ordered = list(curves)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            if first.scan_rate <= 0.0:
                continue
            if abs(second.scan_rate / first.scan_rate - 1.0) > SAME_RATE_TOLERANCE:
                continue
            try:
                early = capacitance(first)[0].farads
                late = capacitance(second)[0].farads
            except CurveError:
                continue
            return DriftCheck(
                rate=first.scan_rate,
                first=early,
                last=late,
                names=(first.name, second.name),
            )
    return None


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
        scan rate. **The order they are given in is taken as the order they
        were measured in**, which is what makes :func:`drift_check` and the
        de-duplication below mean anything. When two of them share a scan
        rate, the later one is the one used for the kinetics and the pair
        becomes the drift check: a repeat is worth more as a control than
        as a second point on a line it is collinear with anyway.
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

    # The repeat is found on the order given, which is the measurement
    # order; everything after this works on the rate-sorted series.
    drift = drift_check(curves)

    unique = unique_rates(curves)
    ordered = sorted(unique, key=lambda c: c.scan_rate)
    study = RateStudy(rates=[c.scan_rate for c in ordered], drift=drift)
    if drift is not None:
        study.warnings.append(str(drift))
    if len(unique) < len(curves):
        study.warnings.append(
            f"de las {len(curves)} curvas, {len(curves) - len(unique)} "
            "repetían una velocidad ya medida; para la cinética se usa la "
            "ÚLTIMA de cada velocidad y la repetición queda como control de "
            "deriva"
        )

    for curve in ordered:
        loop, _ = capacitance(curve)
        study.capacitances.append(loop.farads)
        fade = within_file_fade(curve)
        if fade is not None and abs(fade[0]) > 0.5:
            study.warnings.append(f"{1e3 * curve.scan_rate:g} mV/s: {fade[1]}")

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
    study.leakage = leakage_from_rates(ordered)
    study.current_laws = compare_current_laws(ordered)
    study.full_model = fit_full_current(ordered)
    undecided = [law for law in study.current_laws if not law.decisive]
    if undecided and study.leakage is not None:
        worst = max(undecided, key=lambda law: abs(law.leakage_a))
        total = len(study.current_laws)
        head = (
            f"en {len(undecided)} de los {total} potenciales examinados las "
            f"TRES leyes de i(ν) ajustan igual de bien: «i = a·ν^b», la de "
            f"Dunn y «i = k_cap·ν + i_par». Con {len(ordered)} velocidades "
            f"sobre un factor {span:.0f} y dos parámetros no hay con qué "
            f"elegir ahí, así que en esos potenciales el b no es prueba de "
            f"difusión: es compatible con ella y también con una corriente "
            f"constante, que aquí no es una hipótesis — está medida aparte, "
            f"⟨i_par⟩ = {1e6 * study.leakage.leakage_a:.2f} µA con R² = "
            f"{study.leakage.r_squared:.3f} a partir de ΔQ, y la ley de fuga "
            f"pide {1e6 * worst.leakage_a:.2f} µA en {worst.potential:+.3f} V"
        )
        if len(undecided) >= 0.5 * total:
            head += (
                ". Al ser la mitad o más de la ventana, una clasificación de "
                "mecanismo que se apoye en el b dirá «tipo batería» por un "
                "proceso irreversible y no por almacenamiento"
            )
        elif study.full_model:
            resolved = sum(
                1 for model in study.full_model if model.determined()["k_diff"]
            )
            head += (
                f". En el resto la serie SÍ distingue, y entonces la pregunta "
                f"correcta no es cuál de las tres sino las tres a la vez: el "
                f"ajuste de {len(ordered)} velocidades a "
                f"«k_cap·ν + k_diff·√ν + i_par» determina el término difusivo "
                f"en {resolved} de los {len(study.full_model)} potenciales"
            )
        study.warnings.append(head)
    if study.leakage is not None and study.leakage.leakage_a > 0.0:
        efficiencies = [coulombic_efficiency(c) for c in ordered]
        worst = min((e for e in efficiencies if e is not None), default=None)
        if worst is not None and worst < 1.0 - BALANCE_TOLERANCE:
            study.warnings.append(
                f"la eficiencia coulómbica baja a {worst:.2f} en esta serie, y "
                f"el ajuste ΔQ = Q_ox − Q_red = ⟨i_par⟩·t da ⟨i_par⟩ = "
                f"{1e6 * study.leakage.leakage_a:.2g} µA con R² = "
                f"{study.leakage.r_squared:.3f}. Una corriente parásita "
                f"constante aporta más carga cuanto más dura el ciclo, así "
                f"que imita una dependencia con ν: la retención de arriba la "
                f"lleva dentro. Cita la capacitancia de la carga de "
                f"REDUCCIÓN (reversible_capacitance), que la fuga anódica no "
                f"contamina"
            )
    return study


#: Which half of the cycle the Dunn separation is run on.
#:
#: The anodic sweep alone is the usual published choice and is the
#: default, but it is a CHOICE and not a fact about the method. On a
#: material whose oxidation and reduction are not mirror images — most
#: pseudocapacitors — the two halves give different k₁ and k₂, and that
#: difference is information rather than a nuisance. ``"media"`` takes
#: the anodic branch, ``"catodica"`` the cathodic one, and ``"ciclo"``
#: uses ``|i|`` averaged over both, which is what to use when the
#: question is about the electrode rather than about one direction.
SWEEP_CHOICES: tuple[tuple[str, str], ...] = (
    ("media", "Media: sólo la rama anódica"),
    ("catodica", "Media: sólo la rama catódica"),
    ("ciclo", "Ciclo entero: |i| promediado en las dos ramas"),
)


def _current_at(curve: Voltammogram, potential: float,
                sweep: str = "media") -> Optional[float]:
    """Current at one potential, interpolated on the chosen branch.

    Of the REPORTING CYCLE, like every capacitance in this module. On a
    multi-cycle file ``sweeps()`` splits the whole concatenated record,
    so a branch holds one value per cycle at each potential and the
    interpolation lands on whichever the sort put first. That is not a
    catastrophe — the cycles of one file are similar — but it means the
    b-value and the Dunn separation were read off a different cycle from
    the capacitance they are quoted beside, and on a file whose cycles
    fade (2.5 % per cycle at 10 mV/s on the user's) the difference is
    systematic and invisible.
    """
    try:
        curve, _ = curve.reporting_cycle()
        anodic, cathodic = curve.sweeps()
    except CurveError:
        return None

    def on(branch) -> Optional[float]:
        order = np.argsort(branch.potential)
        x, y = branch.potential[order], branch.current[order]
        if x.size < 2 or potential < x[0] or potential > x[-1]:
            return None
        return float(np.interp(potential, x, y))

    if sweep == "catodica":
        return on(cathodic)
    if sweep == "ciclo":
        # |i| on both branches, averaged. Taking the signed mean instead
        # would cancel a symmetric capacitive current to zero, which is
        # the one quantity this separation is about.
        forward, backward = on(anodic), on(cathodic)
        values = [abs(v) for v in (forward, backward) if v is not None]
        return float(np.mean(values)) if values else None
    return on(anodic)


@dataclass
class DunnAnalysis:
    """Dunn's capacitive/diffusive separation, potential by potential.

    ``i(V, ν) = k₁(V)·ν + k₂(V)·√ν``. The first term is
    surface-controlled — double layer plus anything fast enough not to care
    about diffusion — and the second is diffusion-controlled. Both
    coefficients are fitted independently at each potential, which is what
    makes the **shape** of the capacitive current available and not just a
    percentage: the plot everybody publishes is the capacitive curve drawn
    inside the measured voltammogram, and what it shows is *where* in the
    window the diffusive contribution actually sits.

    What the method assumes, and what to do about it:

    * That there are exactly two contributions and that they scale as ν and
      √ν. A third process, or a diffusive one that is not
      semi-infinite over the whole rate range, is absorbed into whichever
      term fits it better.
    * That the current at one potential belongs to the same process at
      every scan rate. A peak that shifts with scan rate — and a
      battery-like peak always does — breaks that, because at the fastest
      rate the potential is no longer on the peak. This is why the
      separation is reported with the peak shift beside it.

    The fractions are a comparison between samples measured the same way,
    not an absolute. Two laboratories quoting "72 % capacitive" for the
    same powder is a coincidence, not agreement.
    """

    potentials: np.ndarray
    k1: np.ndarray
    """Coefficient of ν, in A/(V/s) — a capacitance, potential by potential."""
    k2: np.ndarray
    """Coefficient of √ν, in A/(V/s)^½."""
    r_squared: np.ndarray
    """Quality of the two-term fit at each potential. Where this is poor
    the separation at that potential means nothing."""
    fractions: dict[float, float] = field(default_factory=dict)
    sweep: str = "media"
    """Which branch the currents were read from. Reported because the two
    halves of a cycle give different coefficients on any material whose
    oxidation and reduction are not mirror images."""
    k1_anodic: Optional[np.ndarray] = None
    k2_anodic: Optional[np.ndarray] = None
    k1_cathodic: Optional[np.ndarray] = None
    k2_cathodic: Optional[np.ndarray] = None
    """Coefficients for each branch separately, whatever :attr:`sweep`
    reports. The published figure needs both: it shades the
    surface-controlled region BETWEEN the two capacitive curves, and what
    is left between each of them and the measured trace is the diffusive
    part. Drawing only one half, as this section did at first, leaves a
    figure nobody in the field recognises."""

    def branch_currents(self, scan_rate: float):
        """``(capacitive_anodic, capacitive_cathodic)`` in amps.

        ``None`` when the separation was not run on both branches, which
        happens on a linear sweep that has no return.
        """
        if self.k1_anodic is None or self.k1_cathodic is None:
            return None, None
        rate = float(scan_rate)
        return self.k1_anodic * rate, self.k1_cathodic * rate
    """Capacitive fraction of the total charge, by scan rate."""
    rates: tuple[float, ...] = ()
    warnings: list[str] = field(default_factory=list)

    def capacitive_inside(
        self, curve: Voltammogram
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(anódica, catódica, excedida)`` for the published figure.

        The capacitive curves of both branches at this curve's scan rate,
        each clipped to the current actually measured there, plus the
        mask of where the clipping had to happen.

        The clipping is not cosmetic and it is not a correction. The
        separation says the measured current is the sum of two
        non-negative parts, so a capacitive part larger than the whole is
        the model failing at that potential; drawn unclipped it shades a
        region outside the voltammogram, which is the first thing anyone
        notices about the figure and the last thing the percentages
        admit. Clipped and masked, the figure stays readable and still
        says where the separation did not hold.

        On the user's files the mask is almost entirely the potentials
        where the branch's own current changes sign between one scan rate
        and the next — the cathodic sweep carries anodic current near the
        positive limit — and there the separation is undefined rather
        than wrong.
        """
        anodic, cathodic = self.branch_currents(curve.scan_rate)
        if anodic is None or cathodic is None:
            empty = np.zeros(self.potentials.size, dtype=bool)
            return self.capacitive_current(curve.scan_rate), None, empty
        exceeded = np.zeros(self.potentials.size, dtype=bool)
        out = []
        for capacitive, branch in ((anodic, "media"), (cathodic, "catodica")):
            measured = np.array(
                [_current_at(curve, float(potential), branch)
                 for potential in self.potentials], dtype=float
            )
            with np.errstate(invalid="ignore"):
                over = ~(np.abs(capacitive) <= np.abs(measured))
                over |= np.sign(capacitive) != np.sign(measured)
            over &= np.isfinite(measured)
            exceeded |= over
            clipped = np.where(over, measured, capacitive)
            out.append(np.where(np.isfinite(clipped), clipped, capacitive))
        return out[0], out[1], exceeded

    def capacitive_current(self, scan_rate: float) -> np.ndarray:
        """The capacitive part of the voltammogram at one scan rate, in A.

        This is the curve to draw inside the measured one.
        """
        return self.k1 * float(scan_rate)

    def diffusive_current(self, scan_rate: float) -> np.ndarray:
        """The diffusion-controlled part at one scan rate, in A."""
        return self.k2 * math.sqrt(float(scan_rate))

    #: What the separation is NOT, stated every time it is reported.
    #:
    #: This is the most misread result in the whole field and the
    #: misreading always goes the same way: a voltammogram with obvious
    #: redox peaks comes back "99 % capacitive" and the peaks look like
    #: they have been swallowed. They have not. Dunn separates
    #: SURFACE-CONTROLLED from DIFFUSION-CONTROLLED current, and a
    #: surface-confined redox reaction is surface-controlled — its
    #: current scales with ν, which is what "pseudocapacitive" means — so
    #: it belongs in k₁ by definition and not by accident. It is the same
    #: point the b-value carries: b ≈ 1 does not distinguish double layer
    #: from pseudocapacitive, because both give it.
    CAVEAT = (
        "esta separación es capacitivo frente a DIFUSIVO, no doble capa "
        "frente a redox. Un proceso redox confinado en la superficie es "
        "superficial: su corriente escala con ν, que es lo que significa "
        "«pseudocapacitivo», así que entra en k₁ por definición. Que un "
        "voltamperograma con picos redox evidentes salga «99 % capacitivo» "
        "es el resultado correcto, no un pico que se haya perdido. Lo que "
        "separa doble capa de pseudocapacitivo es la PRESENCIA de picos "
        "redox en el CV, y eso lo decide la clasificación de mecanismo"
    )

    def summary(self) -> str:
        branch = dict(SWEEP_CHOICES).get(self.sweep, self.sweep)
        lines = [f"Separación de Dunn (i = k₁ν + k₂√ν) — {branch}:", ""]
        for rate, fraction in sorted(self.fractions.items()):
            lines.append(f"  {rate * 1e3:7.1f} mV/s   capacitivo "
                         f"{100 * fraction:5.1f} %   difusivo "
                         f"{100 * (1 - fraction):5.1f} %")
        if self.r_squared.size:
            poor = int(np.count_nonzero(self.r_squared < 0.95))
            if poor:
                lines.append("")
                lines.append(
                    f"  en {poor} de {self.r_squared.size} potenciales el "
                    "ajuste de dos términos no describe los datos (R² < 0.95)"
                )
        lines.append("")
        lines.append("  " + self.CAVEAT)
        if self.warnings:
            lines.append("")
            lines.extend("  ⚠ " + text for text in self.warnings)
        return "\n".join(lines)


def dunn_analysis(
    curves: Sequence[Voltammogram],
    window: Optional[tuple[float, float]] = None,
    points: int = 60,
    sweep: str = "media",
    nonnegative: bool = True,
) -> DunnAnalysis:
    """Separate capacitive from diffusive current, Dunn's way.

    Parameters
    ----------
    curves:
        Three or more voltammograms of the same electrode at different scan
        rates. Three is the minimum for a two-parameter fit to mean
        anything at all; five over a decade is what the method needs.
    window:
        Potential range. Defaults to the range common to every curve —
        which is the only honest default, since a potential one curve did
        not reach cannot enter a fit across curves.
    points:
        How many potentials to separate at.
    sweep:
        Which branch to read the current from; see :data:`SWEEP_CHOICES`.
    nonnegative:
        Hold both coefficients at or above zero, which is the default and
        what the method means. Neither contribution can be negative — a
        capacitive current that subtracts from the measured one is not a
        capacitive current — and an unconstrained fit that returns one
        draws a shaded band OUTSIDE the voltammogram it is supposed to sit
        inside. That is what the published figure makes obvious and the
        percentages hide. Set it false only to see the unconstrained
        coefficients.

    Returns
    -------
    DunnAnalysis
    """
    if len(curves) < 3:
        raise CurveError(
            f"la separación de Dunn ajusta dos coeficientes por potencial y "
            f"se han dado {len(curves)} velocidades. Con dos, el ajuste pasa "
            "exactamente por los dos puntos y el reparto es el que quieras"
        )
    ordered = sorted(curves, key=lambda item: item.scan_rate)
    rates = np.array([curve.scan_rate for curve in ordered], dtype=float)
    if np.any(rates <= 0):
        raise CurveError("hay velocidades de barrido nulas o negativas")
    if window is None:
        low = max(curve.window[0] for curve in ordered)
        high = min(curve.window[1] for curve in ordered)
    else:
        low, high = float(min(window)), float(max(window))
    if high <= low:
        raise CurveError(
            "las curvas no comparten ninguna ventana de potencial"
        )

    potentials = np.linspace(low, high, int(points))
    root = np.sqrt(rates)
    k1 = np.zeros(potentials.size)
    k2 = np.zeros(potentials.size)
    quality = np.full(potentials.size, np.nan)
    capacitive = np.zeros((len(ordered), potentials.size))
    total = np.zeros_like(capacitive)

    crossings: list[float] = []

    def separate(branch: str):
        """``(k₁, k₂, R², medidas, k₁ y k₂ sin restringir)``."""
        one = np.zeros(potentials.size)
        two = np.zeros(potentials.size)
        free = np.zeros((2, potentials.size))
        uncertainty = np.full((2, potentials.size), np.inf)
        fit_quality = np.full(potentials.size, np.nan)
        measured = np.zeros((len(ordered), potentials.size))
        for column, potential in enumerate(potentials):
            currents = [_current_at(curve, potential, branch) for curve in ordered]
            if any(value is None for value in currents):
                continue
            observed = np.array(currents, dtype=float)
            # i/√ν = k₁√ν + k₂ — a straight line in √ν, which is the form
            # that keeps the fit linear and the two coefficients separable.
            if nonnegative:
                # The fit is on |i| and the branch's own sign is put back
                # afterwards. Constraining the SIGNED current would make
                # the whole cathodic branch illegal, and the sign of a
                # capacitive current is the sign of the sweep, not a
                # property of the coefficient.
                direction = float(np.sign(np.median(observed))) or 1.0
                if np.any(np.sign(observed) != direction):
                    crossings.append(float(potential))
                magnitude = np.abs(observed)
                design = np.vstack([root, np.ones_like(root)]).T
                # Non-negativity, and nothing else. The model also
                # implies k₁ν ≤ |i_ν| at every rate, and imposing THAT as
                # a ceiling looks right and is not: the minimum of a few
                # noisy |i_ν|/ν sits below the true k₁, so the ceiling
                # binds on noise and clips k₁ downwards. Measured on the
                # synthetic pure pseudocapacitor, where the answer is
                # 100 % capacitive by construction, it returned 91 %.
                # Keeping the capacitive curve inside the measured loop is
                # the FIGURE's job — see `capacitive_inside` — because
                # there the excess can be shown rather than absorbed.
                solution = _lsq_linear(design, magnitude / root,
                                       bounds=([0.0, 0.0], [np.inf, np.inf]))
                coefficients = solution.x
                slope, intercept = (float(coefficients[0]) * direction,
                                    float(coefficients[1]) * direction)
                # Kept so the warning below can use the criterion this
                # package already calibrated: a coefficient at its bound
                # only means something when the UNRESTRICTED fit wanted
                # to go appreciably negative. On an ideal capacitor the
                # diffusive term is genuinely zero and half the
                # potentials land on the bound, which is the right
                # answer and nothing to complain about.
                # ...with the standard errors of the UNCONSTRAINED fit,
                # because "negative by more than 5 % of its own scale"
                # compares noise with noise when the coefficient is
                # genuinely zero: on an ideal capacitor the diffusive
                # term IS zero, so the scale is itself a noise
                # excursion and a fixed fraction of it is met by chance.
                # Reading one cycle instead of four made that visible —
                # 19 of 120 coefficients on a clean synthetic capacitor.
                # What a coefficient has to beat is its own uncertainty.
                shape = np.vstack([root, np.ones_like(root)]).T
                target = magnitude / root
                loose, *_ = np.linalg.lstsq(shape, target, rcond=None)
                spare = shape.shape[0] - 2
                leftover = target - shape @ loose
                if spare > 0:
                    scatter = float((leftover**2).sum()) / spare
                    try:
                        spread = scatter * np.linalg.inv(shape.T @ shape)
                        sigma = np.sqrt(np.maximum(np.diag(spread), 0.0))
                    except np.linalg.LinAlgError:
                        sigma = np.full(2, np.inf)
                else:
                    sigma = np.full(2, np.inf)
                free[:, column] = loose
                uncertainty[:, column] = sigma
                predicted = (slope * rates + intercept * root) * direction
                total_variation = float(
                    ((magnitude - magnitude.mean()) ** 2).sum()
                )
                residual = float(((magnitude - predicted) ** 2).sum())
                r2 = (1.0 - residual / total_variation
                      if total_variation > 0 else 1.0)
            else:
                slope, intercept, _, _ = _fit_line(root, observed / root)
                # R² on the CURRENT, not on i/√ν, so that the constrained
                # and unconstrained fits are judged by the same number.
                # The transformed variable has far less variance to
                # explain, and its R² reads as a much worse fit than the
                # same model has: on the user's files, 46 of 60 potentials
                # below 0.95 against 4 of 60.
                predicted = slope * rates + intercept * root
                spread = float(((observed - observed.mean()) ** 2).sum())
                residual = float(((observed - predicted) ** 2).sum())
                r2 = 1.0 - residual / spread if spread > 0 else 1.0
            one[column], two[column], fit_quality[column] = slope, intercept, r2
            measured[:, column] = observed
        return one, two, fit_quality, measured, free, uncertainty

    # Both branches, always. The `sweep` choice decides which one the
    # headline fractions report, but the FIGURE everybody publishes needs
    # both: it fills the surface-controlled region between the cathodic
    # and anodic capacitive curves, and the diffusive part is what is left
    # between each of those and the measured trace.
    k1_anodic, k2_anodic, *_ = separate("media")
    k1_cathodic, k2_cathodic, *_ = separate("catodica")
    k1, k2, quality, total, unrestricted, spread = separate(sweep)
    pinned = 0
    if nonnegative:
        # A one-sided test at each of 2 x points coefficients, with the
        # level divided by how many tests are being made. Without that
        # correction the warning fires on noise: a 2-sigma rule applied
        # 120 times on a synthetic capacitor, where the diffusive term
        # is zero by construction, reported 7 "genuinely negative"
        # coefficients. It is the same correction the Raman peak finder
        # makes, for the same reason.
        tests = max(1, 2 * potentials.size)
        spare = max(1, len(ordered) - 2)
        threshold = float(_student_t.ppf(1.0 - NEGATIVE_LEVEL / tests, spare))
        for coefficients, sigma in zip(unrestricted, spread):
            usable = np.isfinite(sigma) & (sigma > 0.0)
            pinned += int(np.count_nonzero(
                usable & (coefficients < -threshold * sigma)))
    for index in range(len(ordered)):
        capacitive[index] = k1 * rates[index]

    fractions: dict[float, float] = {}
    for index, curve in enumerate(ordered):
        denominator = float(np.abs(total[index]).sum())
        if denominator <= 0.0:
            continue
        fractions[curve.scan_rate] = float(
            min(1.0, np.abs(capacitive[index]).sum() / denominator)
        )

    warnings: list[str] = []
    span = float(rates.max() / rates.min())
    if span < MIN_RATE_SPAN:
        warnings.append(
            f"las velocidades abarcan un factor {span:.1f}. Separar ν de √ν "
            "sobre menos de una década es pedirle a un ajuste que distinga "
            "dos funciones que en ese tramo se parecen mucho"
        )
    # A coefficient that is negative by a hair is a coefficient that is
    # zero: on an ideal capacitor the diffusive term is genuinely zero and
    # the noise puts it below zero at half the potentials. Only a
    # coefficient negative by an appreciable fraction of its own scale is
    # evidence that the two-term model does not fit.
    if nonnegative:
        # A coefficient driven onto its bound is the constrained fit's way
        # of saying what the unconstrained one said by going negative: the
        # two-term model does not describe the current there. The
        # difference is that the figure stays readable — the shaded band
        # stays inside the voltammogram — instead of a negative k₁ drawing
        # a capacitive current on the wrong side of the loop.
        unique_crossings = sorted(set(round(value, 3) for value in crossings))
        if unique_crossings:
            warnings.append(
                f"en {len(unique_crossings)} potenciales la corriente de la "
                f"rama CAMBIA DE SIGNO entre una velocidad y otra "
                f"({unique_crossings[0]:+.3f} a {unique_crossings[-1]:+.3f} V). "
                "Ahí la separación no está definida: k₁ν es una fracción de "
                "una corriente que no tiene un signo, y la banda capacitiva "
                "de la figura puede salirse del lazo por ese motivo y no por "
                "el ajuste. Es la firma de un cruce por cero que se mueve "
                "con la velocidad — en estos archivos, la rama catódica "
                "lleva corriente anódica cerca del límite positivo"
            )
        if pinned:
            warnings.append(
                f"{pinned} de los {2 * potentials.size} coeficientes se han "
                "quedado en CERO porque el ajuste SIN restringir los quería "
                "negativos POR MÁS DE DOS VECES SU PROPIA INCERTIDUMBRE, y "
                "ninguna de las dos contribuciones puede serlo: en esos "
                "modelo no describe la corriente. Suele ser un pico que se "
                "desplaza con la velocidad —a la velocidad más alta ese "
                "potencial ya no está sobre el pico— o una corriente que no "
                "escala ni con ν ni con √ν, como una parásita constante"
            )
    else:
        negative = 0
        for coefficients in (k1, k2):
            scale = (float(np.max(np.abs(coefficients)))
                     if coefficients.size else 0.0)
            if scale > 0:
                negative += int(np.count_nonzero(coefficients < -0.05 * scale))
        if negative:
            warnings.append(
                f"{negative} coeficientes salen negativos de verdad (más de "
                "un 5 % de su propia escala) y ninguna de las dos "
                "contribuciones puede serlo: ahí el modelo de dos términos "
                "no describe la corriente"
            )
    if sweep not in {key for key, _ in SWEEP_CHOICES}:
        raise CurveError(
            f"rama desconocida {sweep!r}; las que hay son: "
            + ", ".join(key for key, _ in SWEEP_CHOICES)
        )
    return DunnAnalysis(
        potentials=potentials, k1=k1, k2=k2, r_squared=quality,
        fractions=fractions, rates=tuple(rates), warnings=warnings,
        sweep=sweep,
        k1_anodic=k1_anodic, k2_anodic=k2_anodic,
        k1_cathodic=k1_cathodic, k2_cathodic=k2_cathodic,
    )


def _dunn_split(
    curves: Sequence[Voltammogram], low: float, high: float
) -> dict[float, float]:
    """Capacitive fraction by scan rate — the summary of :func:`dunn_analysis`."""
    if len(curves) < 3:
        return {}
    try:
        return dunn_analysis(curves, window=(low, high), points=40).fractions
    except CurveError:
        return {}


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
    _, outer_intercept, _, _ = _fit_line(1.0 / np.sqrt(rates), values)
    _, total_intercept, _, _ = _fit_line(np.sqrt(rates), 1.0 / values)
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
            _, cathodic = curve.sweeps()
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
    "DunnAnalysis",
    "RateStudy",
    "RedoxPeak",
    "analyse_cv",
    "analyse_rate_study",
    "capacitance",
    "dunn_analysis",
    "find_redox_peaks",
    "integrate_charge",
]
