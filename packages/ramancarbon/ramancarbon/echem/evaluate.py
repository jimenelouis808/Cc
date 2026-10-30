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

from .curve import CurveError, Voltammogram, load_echem_database
from .cv import PARASITIC_SHARE, CVResult, RateStudy
from .gcd import GCDResult

#: Minimum decades of current density for a Tafel fit to be determined.
MIN_TAFEL_DECADES = 1.0

#: Tafel slopes above this (mV/dec) usually mean transport, not kinetics.
TAFEL_TRANSPORT_LIMIT = 200.0

#: An exchange current density below this (mA/cm²) is not a measurement.
#: It is what an extrapolation over many decades produces when the
#: overpotential axis is offset — almost always a wrong reference electrode
#: or a wrong pH, either of which shifts the whole axis by a volt or so.
IMPLAUSIBLE_J0 = 1e-14


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

    # A b-value below 1 is only evidence of diffusion if diffusion is
    # what put it there. A constant parasitic current — corrosion, carbon
    # oxidation, electrolyte breakdown at the positive limit — dilutes a
    # capacitive current by more the slower the scan, and over the decade
    # of scan rates anyone actually measures that is indistinguishable
    # from a sqrt(nu) term. So when the rate study found the three laws
    # tied AND measured a leak independently from the charge balance, the
    # b-value is withdrawn as evidence rather than being believed.
    #
    # On the user's files this is the difference between the right answer
    # and the wrong one: b = 0.53 at the positive limit, a coulombic
    # efficiency of 0.48, and a measured 0.41 uA of leak. Believing the b
    # classified a leaky capacitor as a battery electrode and told the
    # user to report it in mAh/g.
    caveats: list[str] = []
    leaky = False
    bent_by = 0.0
    if rates is not None and rates.full_model and rates.b_values:
        # The DIRECT test, available once the series is wide enough to
        # fit all three terms at once: at the potential where b is
        # lowest — the one the classification rests on — is there a
        # determined parasitic current, and is it a real share of the
        # current there? Counting how often the leak law "wins or ties"
        # against the other two, which is what this used to do, is a
        # proxy for that question and a poor one: it goes to zero as
        # soon as the series is wide enough to tell the laws apart,
        # exactly when the parasitic current is best measured.
        worst = min(rates.b_values, key=lambda value: value.b)
        nearest = min(rates.full_model,
                      key=lambda m: abs(m.potential - worst.potential))
        slowest = min(rates.rates) if rates.rates else 0.0
        # Magnitudes, not the signed sum: where the fitted constant has
        # the opposite sign to the other two the sum nearly cancels and
        # the share blows up. On the synthetic hybrid, which has no
        # parasitic current at all, that read 92 %.
        total = (abs(nearest.capacitance_f * slowest)
                 + abs(nearest.diffusive * math.sqrt(slowest))
                 + abs(nearest.leakage_a))
        # And the fit alone is not enough. A constant that the i(ν) fit
        # likes must also show up where it cannot hide: in the charge
        # balance, which measures the irreversible current without
        # looking at how anything scales with ν. Both, or neither.
        balance = rates.leakage
        corroborated = (
            balance is not None
            and balance.r_squared > 0.9
            and max(balance.shares.values(), default=0.0) > 0.1
        )
        if nearest.determined()["i_par"] and total > 0.0 and corroborated:
            bent_by = abs(nearest.leakage_a) / total
            leaky = bent_by > PARASITIC_SHARE
    elif rates is not None and rates.current_laws and rates.leakage is not None:
        # The fallback for a series too narrow to fit three terms: the
        # leak law has to be at least as good as the other two — it
        # either wins or ties — at half the potentials or more.
        standing = [
            law for law in rates.current_laws
            if law.best == "fuga" or not law.decisive
        ]
        share = max(rates.leakage.shares.values(), default=0.0)
        leaky = (
            len(standing) >= 0.5 * len(rates.current_laws)
            and rates.leakage.leakage_a > 0.0
            and rates.leakage.r_squared > 0.9
            and share > 0.1
        )
    if leaky:
        evidence["battery"] = [
            text for text in evidence["battery"] if not text.startswith("b = ")
        ]

    if rates is not None and rates.b_values and leaky:
        sources += 1
        b = float(min(value.b for value in rates.b_values))
        if bent_by > 0.0:
            caveats.append(
                f"el b baja a {b:.2f} y no se toma como prueba de difusión: "
                f"en ese mismo potencial el ajuste de los tres términos "
                f"—k_cap·ν + k_diff·√ν + i_par— determina una corriente "
                f"parásita que es el {100 * bent_by:.0f} % de la corriente "
                f"total a la velocidad más lenta. Una constante dobla el b "
                f"hacia abajo sin que haya difusión de por medio, así que "
                f"con esa fracción el b mide las dos cosas a la vez. Lo que "
                f"SÍ es una medida del término difusivo es k_diff, que va "
                f"aparte y con su incertidumbre"
            )
        else:
            caveats.append(
                f"el b baja a {b:.2f}, pero no cuenta como difusión: en esta "
            f"serie las tres leyes de i(ν) ajustan igual de bien y hay "
            f"{1e6 * rates.leakage.leakage_a:.2f} µA de corriente parásita "
            f"medidos aparte, suficientes para explicar por sí solos que b "
            f"se aparte de 1. Para decidirlo hace falta quitar la fuga "
                f"—ventana más estrecha, electrolito desoxigenado— y repetir"
            )
    elif rates is not None and rates.b_values:
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
    ] + caveats
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


#: Current densities the overpotential is reported at, in mA/cm².
#:
#: Ten is the convention — it comes from a 10 %-efficient solar cell
#: under one sun — and it is not enough on its own: a catalyst that is
#: excellent at 10 and collapses at 100 is one nobody can use, and only
#: the pair says so. Fifty and a hundred are what the field asks for now.
BENCHMARK_DENSITIES: tuple[float, ...] = (10.0, 50.0, 100.0)

#: Current density, in mA/cm², above which the numbers are not physical.
#:
#: A very good laboratory catalyst on a very good support reaches a few
#: thousand mA/cm² in a flow cell; a beaker with a glassy-carbon disc does
#: not get near it. So a curve that claims more than this is not reporting
#: a remarkable electrode, it is reporting amps taken for milliamps or the
#: wrong area — and the mistake is worth catching here because the iR
#: correction is multiplied by the same factor and the overpotential with
#: it. Seen: an axis running to 2·10⁵ mA/cm² and an overpotential of
#: 400 V, from a measurement whose real maximum was 240 mA/cm².
ABSURD_CURRENT_DENSITY = 1.0e4

#: Overpotential, in volts, below which a NEGATIVE onset is reported as
#: an impossibility rather than as a number.
#:
#: An overpotential is defined against the thermodynamic potential of the
#: reaction, so a negative one says the reaction is running before
#: thermodynamics allows it. That cannot happen, and the small negative
#: values a real measurement produces are the resolution of the
#: potential axis rather than a claim: a couple of millivolts is the
#: reference electrode being read to the millivolt. Beyond that it is an
#: OFFSET, and the offset travels into the benchmark overpotential, the
#: Tafel slope and the exchange current alike.
ONSET_SIGN_TOLERANCE = 0.005

#: Fraction of the measured overpotential that an ohmic drop may reach
#: before the overpotential stops being a measurement of the catalyst.
#:
#: The iR drop is subtracted from the measured overpotential, so one
#: comparable to it leaves a difference of two large numbers. At a third
#: the correction is already the dominant term in any comparison with
#: the literature.
IR_SHARE_LIMIT = 0.33

#: Overpotential at the benchmark, in volts, below which a NON-NOBLE
#: catalyst is reported as suspect rather than as excellent.
#:
#: Platinum in acid sits at 20-35 mV with a Tafel slope near 30 mV/dec,
#: and the commonest way a non-noble electrode reaches that number is
#: not synthesis: it is platinum DISSOLVED from the counter electrode
#: and re-deposited on the working one. It is the best documented
#: artefact in the whole HER literature, it takes nanograms to show, and
#: it is invisible in every electrochemical measurement because what is
#: being measured afterwards really is platinum. Fifty millivolts sits
#: above the published range for platinum and below where the good
#: non-noble catalysts of this class report.
PLATINUM_SUSPICION_V = 0.050

#: Cathodic current density, in mA/cm², that counts as a sustained
#: background when it flows POSITIVE of the reaction's own potential.
#:
#: Nothing should: positive of the equilibrium the reaction cannot run.
#: A steady cathodic current there is something else being reduced, and
#: in an aqueous cell that was not purged it is dissolved oxygen, whose
#: diffusion-limited current on a millimetre disc in air-saturated acid
#: is a few tenths of a mA/cm². That current adds to the sweep, so the
#: curve reaches any chosen density earlier than the reaction alone
#: would and every overpotential read from it comes out too small.
BACKGROUND_CURRENT_LIMIT = 0.05


#: Fraction of a hold's own length that counts as settling rather than
#: as performance.
#:
#: A freshly wetted electrode moves for the first minutes whatever it is
#: going to do afterwards, and a drift rate fitted through that start
#: measures the wetting. A tenth is short enough to keep a short hold
#: usable and long enough to clear the initial transient.
STABILITY_SETTLING = 0.10

#: Overpotential gained per hour, in volts, above which a hold is a
#: FAILING electrode rather than a drifting one.
#:
#: Ten millivolts an hour is already a hundred over a ten-hour test,
#: which is the length everybody quotes. Above fifty the electrode will
#: not finish that test at all, and saying so is more useful than a
#: retention percentage computed over whatever fraction of it ran.
STABILITY_DRIFT_V_PER_H = 0.010
STABILITY_FAILING_V_PER_H = 0.050


@dataclass
class StabilityResult:
    """What a constant-current hold did to the overpotential."""

    current_ma_cm2: float
    duration_h: float
    overpotential_start: Optional[float] = None
    """η once the first :data:`STABILITY_SETTLING` of the hold is past,
    in volts. Not the very first point, which is the cell wetting."""
    overpotential_end: Optional[float] = None
    drift_v_per_h: Optional[float] = None
    """Slope over the settled part. The number that decides whether the
    electrode survives the test everybody quotes."""
    r_squared: float = 0.0
    hours_to_lose_100mv: Optional[float] = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Estabilidad a {self.current_ma_cm2:.3g} mA/cm² durante "
                 f"{self.duration_h:.2f} h"]
        if self.overpotential_start is not None:
            lines.append(
                f"  η: {1e3 * self.overpotential_start:.1f} → "
                f"{1e3 * self.overpotential_end:.1f} mV "
                f"({1e3 * (self.overpotential_end - self.overpotential_start):+.1f} mV)")
        if self.drift_v_per_h is not None:
            lines.append(f"  deriva: {1e3 * self.drift_v_per_h:+.1f} mV/h "
                         f"(R² = {self.r_squared:.4f})")
        if self.hours_to_lose_100mv is not None:
            lines.append(f"  a ese ritmo, 100 mV más en "
                         f"{self.hours_to_lose_100mv:.1f} h")
        for warning in self.warnings:
            lines.append(f"  ⚠ {warning}")
        return "\n".join(lines)


def stability(
    time_s: Sequence[float],
    overpotential_v: Sequence[float],
    current_ma_cm2: float,
) -> StabilityResult:
    """A constant-current hold, read as a drift rate rather than a number.

    An overpotential and a Tafel slope describe the instant of the
    measurement. What decides whether a catalyst is usable is whether it
    holds, and the field quotes that as "η after 10 h" — which is a
    single number that hides the shape and, worse, invites a hold that
    ran for one hour to be reported as if it had run for ten.

    So what comes back is a **rate**, fitted over the settled part of
    whatever actually ran, plus how long that rate needs to cost another
    hundred millivolts. Extrapolating a measured rate to a time nobody
    measured is stated as the extrapolation it is; it is not quoted as a
    result.
    """
    time = np.asarray(time_s, dtype=float)
    eta = np.asarray(overpotential_v, dtype=float)
    if time.size != eta.size or time.size < 10:
        raise ValueError("hacen falta al menos diez puntos de la retención")
    span = float(time[-1] - time[0])
    if span <= 0:
        raise ValueError("el eje de tiempo no avanza")

    result = StabilityResult(current_ma_cm2=float(current_ma_cm2),
                             duration_h=span / 3600.0)
    settled = time >= time[0] + STABILITY_SETTLING * span
    if np.count_nonzero(settled) < 5:
        settled = np.ones_like(time, dtype=bool)
    result.overpotential_start = float(eta[settled][0])
    result.overpotential_end = float(eta[-1])

    slope, intercept = np.polyfit(time[settled], eta[settled], 1)
    predicted = intercept + slope * time[settled]
    total = float(np.sum((eta[settled] - np.mean(eta[settled])) ** 2))
    residual = float(np.sum((eta[settled] - predicted) ** 2))
    result.r_squared = 1.0 - residual / total if total > 0 else 0.0
    result.drift_v_per_h = float(slope) * 3600.0

    if result.drift_v_per_h > 0:
        result.hours_to_lose_100mv = 0.100 / result.drift_v_per_h

    if result.duration_h < 1.0:
        result.warnings.append(
            f"la retención duró {60 * result.duration_h:.0f} minutos. Lo que "
            "el campo cita es «η tras 10 h», y una deriva medida sobre menos "
            "de una hora extrapolada a diez no es una medida: los "
            "mecanismos de pérdida cambian de pendiente. Lo que se informa "
            "aquí es el RITMO sobre lo que de verdad corrió"
        )
    if result.drift_v_per_h > STABILITY_FAILING_V_PER_H:
        result.warnings.append(
            f"la deriva es de {1e3 * result.drift_v_per_h:.0f} mV/h, así que "
            f"el electrodo gana otros 100 mV en "
            f"{result.hours_to_lose_100mv:.1f} h: **no termina** una prueba "
            "de diez horas a esta corriente. Un sobrepotencial inicial, por "
            "bueno que sea, no describe un electrodo así"
        )
    elif result.drift_v_per_h > STABILITY_DRIFT_V_PER_H:
        result.warnings.append(
            f"la deriva es de {1e3 * result.drift_v_per_h:.0f} mV/h, que son "
            f"{1e3 * 10 * result.drift_v_per_h:.0f} mV sobre una prueba de "
            "diez horas. Hay que informarla junto al sobrepotencial inicial"
        )
    if result.r_squared < 0.9:
        result.warnings.append(
            f"la deriva no es una recta (R² = {result.r_squared:.3f}): el "
            "ritmo de pérdida cambia durante la propia retención, así que "
            "una sola pendiente lo promedia y la extrapolación vale aún "
            "menos. Mira la curva"
        )
    return result


@dataclass
class CatalysisResult:
    """HER or OER figures of merit."""

    reaction: str
    overpotential_at_benchmark: Optional[float] = None
    """η at 10 mA/cm², in volts. The field's comparison point."""
    benchmark_ma_cm2: float = 10.0
    overpotentials: dict[float, Optional[float]] = field(default_factory=dict)
    """η at each of :data:`BENCHMARK_DENSITIES`, in volts, ``None`` where
    the curve does not reach it.

    Ten alone stopped being enough some years ago: a catalyst that is
    excellent at 10 mA/cm² and collapses at 100 is a catalyst nobody can
    use, and the difference between η₁₀ and η₁₀₀ is what says so. The
    ones the measurement did not reach are reported as not reached rather
    than extrapolated — a Tafel line extended two decades past the data
    is a drawing."""
    curve_j: Optional[np.ndarray] = None
    """|j| in mA/cm², iR-corrected, on the active branch."""
    curve_eta: Optional[np.ndarray] = None
    """η in volts, matching :attr:`curve_j`. Carried so the section can
    draw the polarisation curve itself and not only its logarithm: a
    Tafel plot hides the shape, and the shape is where a mass-transport
    limit or a bubble problem shows up."""
    tafel: Optional[TafelResult] = None
    onset_overpotential: Optional[float] = None
    mass_activity_a_per_g: Optional[float] = None
    ecsa_normalised: Optional[float] = None
    conversion_note: str = ""
    ir_note: str = ""
    background_ma_cm2: Optional[float] = None
    """Sustained cathodic current density flowing POSITIVE of the
    reaction's own potential, where nothing should flow. Usually
    dissolved oxygen in a cell that was not purged."""
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Reacción: {self.reaction}"]
        if self.conversion_note:
            lines.append(f"  {self.conversion_note}")
        if self.ir_note:
            lines.append(f"  {self.ir_note}")
        lines.append("")
        for level, value in sorted(self.overpotentials.items()):
            if level == self.benchmark_ma_cm2:
                continue
            lines.append(
                f"  η a {level:g} mA/cm²  : "
                + (f"{1e3 * value:.0f} mV" if value is not None
                   else "la curva no llega a esa densidad")
            )
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
    if 0.0 < result.exchange_current_density < IMPLAUSIBLE_J0:
        result.warnings.append(
            f"la densidad de corriente de intercambio sale "
            f"{result.exchange_current_density:.2g} mA/cm², que no es una "
            "medida de nada: es lo que produce extrapolar muchas décadas "
            "cuando el eje de sobrepotencial está desplazado. Comprueba el "
            "electrodo de referencia y el pH — equivocarse en cualquiera de "
            "los dos mueve el eje entero alrededor de un voltio"
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

    mismatch = curve.metadata.get("aviso_referencia")
    if mismatch:
        result.warnings.append(mismatch)
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

    result.curve_j = j
    result.curve_eta = eta

    def at(level: float) -> Optional[float]:
        """η where the curve reaches ``level``, or ``None``.

        BOTH ends are checked. ``np.interp`` clamps: asked for a density
        below the measured range it returns the first value without
        saying so, and a guard on the upper end alone lets that through.
        On a curve whose active branch starts at 250 mA/cm² — an ordinary
        HER measurement that simply never went slower — that reported the
        SAME overpotential at 10, 50 and 100 mA/cm², and the number it
        reported was the potential at the very end of the sweep.
        """
        if j.size < 2 or not (j.min() <= level <= j.max()):
            return None
        return float(np.interp(level, j, eta))

    result.overpotential_at_benchmark = at(benchmark)
    result.onset_overpotential = at(1.0)
    result.overpotentials = {level: at(level) for level in BENCHMARK_DENSITIES}

    if result.overpotential_at_benchmark is None and j.min() > benchmark:
        result.warnings.append(
            f"la rama activa empieza en {j.min():.0f} mA/cm², por encima de "
            f"los {benchmark:g} del punto de comparación, así que η a esa "
            "densidad no está medido. Extrapolarlo hacia abajo sería dibujar: "
            "mide más despacio o desde un potencial menos activo"
        )
    # An overpotential is measured FROM the thermodynamic potential, so a
    # negative one says the reaction runs before thermodynamics allows
    # it. It cannot, and what it actually reports is an offset in the
    # potential axis -- which every number below inherits.
    onset = result.onset_overpotential
    if onset is not None and onset < -ONSET_SIGN_TOLERANCE:
        result.warnings.append(
            f"el sobrepotencial de inicio sale NEGATIVO ({1e3 * onset:.0f} "
            "mV), y eso es imposible: un sobrepotencial se mide desde el "
            "potencial termodinámico de la reacción, así que uno negativo "
            "dice que la reacción corre antes de que la termodinámica la "
            "permita. Lo que informa de verdad es un DESPLAZAMIENTO del eje "
            "de potencial, y lo heredan todos los números de abajo — el η "
            "del punto de comparación, la pendiente de Tafel y la corriente "
            "de intercambio. Tres causas, en este orden de probabilidad: el "
            "electrodo de referencia no está donde dice la tabla (se "
            "contrasta contra uno nuevo, o contra el potencial de equilibrio "
            "del hidrógeno en un hilo de platino en este mismo electrolito), "
            "el pH no es el nominal, o lo que se está midiendo cerca del "
            "inicio no es la reacción sino una corriente de fondo"
        )

    # What the ohmic drop would be at the benchmark, in millivolts, next
    # to the overpotential it is subtracted from. Without a resistance
    # the useful number is the one that would eat the whole measurement.
    benchmark_eta = result.overpotential_at_benchmark
    if benchmark_eta is not None and electrode.area_cm2:
        current_a = benchmark * 1e-3 * float(electrode.area_cm2)
        resistance = electrode.resistance_ohm
        if resistance:
            share = (current_a * float(resistance)) / abs(benchmark_eta)
            if share > IR_SHARE_LIMIT:
                result.warnings.append(
                    f"a {benchmark:g} mA/cm² circulan {1e3 * current_a:.3g} mA, "
                    f"así que la caída óhmica vale "
                    f"{1e3 * current_a * float(resistance):.0f} mV sobre un "
                    f"sobrepotencial medido de {1e3 * benchmark_eta:.0f} mV: "
                    f"el {100 * share:.0f} %. La corrección no es un detalle "
                    "aquí, es el término dominante, y el η corregido es la "
                    "diferencia de dos números parecidos"
                )
        elif current_a > 0:
            fatal = abs(benchmark_eta) / current_a
            result.warnings.append(
                f"a {benchmark:g} mA/cm² circulan {1e3 * current_a:.3g} mA, "
                f"así que bastarían {fatal:.1f} Ω de resistencia no "
                f"compensada para explicar ENTERO el sobrepotencial de "
                f"{1e3 * benchmark_eta:.0f} mV que se informa. Una celda de "
                "laboratorio con un capilar de Luggin cerca anda por ahí, "
                "así que sin medir la resistencia este número no distingue "
                "un catalizador bueno de una celda con poca caída: mide la "
                "impedancia de esta misma celda y da su corte de alta "
                "frecuencia"
            )

    # A sustained cathodic current POSITIVE of the reaction's own
    # potential. Nothing should flow there, so what does is something
    # else being reduced -- and it adds to the sweep, so every
    # overpotential read from the curve comes out too small.
    before = eta < 0.0
    if np.any(before) and np.count_nonzero(before) >= 5:
        background = float(np.median(j[before]))
        if background > BACKGROUND_CURRENT_LIMIT:
            result.background_ma_cm2 = background
            result.warnings.append(
                f"por encima del potencial de equilibrio circula una "
                f"corriente catódica sostenida de {background:.3g} mA/cm², y "
                "ahí no debería circular ninguna: la reacción no puede correr "
                "antes de su propio potencial. Lo más común en una celda "
                "acuosa sin desairear es la reducción del OXÍGENO disuelto, "
                "cuya corriente límite en un disco de milímetros en ácido al "
                "aire son unas décimas de mA/cm². Esa corriente se SUMA al "
                "barrido, así que la curva llega a cualquier densidad antes "
                "de lo que llegaría la reacción sola y todos los "
                "sobrepotenciales salen pequeños de más. Se quita burbujeando "
                "N₂ o Ar veinte minutos y manteniendo la manta de gas durante "
                "la medida"
            )

    if j.max() > ABSURD_CURRENT_DENSITY:
        result.warnings.append(
            f"la densidad de corriente llega a {j.max():.3g} mA/cm², que son "
            f"{j.max() / 1000:.3g} A/cm². Ningún electrodo de laboratorio da "
            "eso: casi siempre es que la CORRIENTE está en otras unidades de "
            "las que se han supuesto (un factor de mil entre A y mA) o que el "
            "área es la equivocada. Compruébalas antes de leer nada de lo de "
            "abajo, porque la corrección óhmica también se multiplica por ese "
            "factor y el sobrepotencial con ella"
        )

    result.tafel = tafel_analysis(eta, j)

    # Platinum from the counter electrode. The commonest way a
    # non-noble electrode reports platinum's numbers is not synthesis.
    if (reaction == "HER" and benchmark_eta is not None
            and 0.0 < benchmark_eta < PLATINUM_SUSPICION_V):
        slope = (result.tafel.slope_mv_per_decade
                 if result.tafel and result.tafel.valid else None)
        text = (
            f"η a {benchmark:g} mA/cm² sale {1e3 * benchmark_eta:.0f} mV, que "
            "es el intervalo del PLATINO (20-35 mV). Para un catalizador no "
            "noble ese número se explica casi siempre por platino disuelto "
            "del CONTRAELECTRODO y redepositado en el de trabajo: hacen falta "
            "nanogramos, no se ve en ninguna medida electroquímica —porque lo "
            "que se mide después es platino de verdad— y es el artefacto "
            "mejor documentado de la literatura de HER"
        )
        if slope is not None:
            text += (f". La pendiente de Tafel ({slope:.0f} mV/dec) es el "
                     "otro indicio: el platino está cerca de 30")
        text += (". Se descarta repitiendo con un contraelectrodo de GRAFITO "
                 "en celda limpia, o buscando Pt 4f por XPS en el electrodo "
                 "medido")
        result.warnings.append(text)

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
    "IMPLAUSIBLE_J0",
    "MIN_TAFEL_DECADES",
    "TAFEL_TRANSPORT_LIMIT",
    "CatalysisResult",
    "StorageVerdict",
    "TafelResult",
    "analyse_catalysis",
    "classify_storage",
    "tafel_analysis",
]
