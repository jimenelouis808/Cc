"""Identifying elements in a survey scan.

A survey is a list of peaks and a question: which elements are on this
surface? The wrong way to answer it is to take each peak, look up the
nearest tabulated line, and print the element's name — that finds selenium
in every spectrum, because at ±2 eV over a 1400 eV range almost every
position is near *something*.

So the rule here is the same one the Raman side uses for secondary phases
and the XRD side for unknown phases: **an element is a spectrum, not a
line.** To be identified, an element must show its strongest line *and* at
least one other feature — a second core level, or its Auger group. An
element whose only other feature falls outside the measured range is
reported as uncorroborated, with that as the stated reason, instead of
being quietly promoted or quietly dropped.

Two things make this harder in XPS than the analogous problem elsewhere,
and both are handled explicitly:

**Auger lines move with the anode.** They sit at fixed *kinetic* energy, so
their apparent binding energy is ``hν − KE − φ``. With aluminium the nickel
LMM group lands on top of the iron 2p region; with magnesium it does not.
Without a photon energy no Auger group can be placed at all, and this module
says which identifications were weakened by that rather than pretending the
Auger lines are not there.

**Lines overlap.** Iron 3p and selenium 3d are both near 55 eV, which is
exactly the pair that matters in an iron-selenide-decorated carbon. When
one peak is the only evidence for two elements, both are told, and the
report says which other region settles it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from scipy.signal import find_peaks as signal_find_peaks

from ..core.baseline import snip_baseline
from .elements import XPSDatabase, load_xps_database
from .spectrum import XPSError, XPSSpectrum

#: How far a peak may sit from a tabulated photoemission line, in eV.
#: Generous: a survey is measured at 0.5–1 eV per point and an uncorrected
#: charge shift of a couple of eV is normal on a survey, which is usually
#: taken before anything is referenced.
LINE_TOLERANCE = 2.0

#: Minimum prominence, in units of the local counting noise, for a feature
#: to be a peak at all. See :func:`find_survey_peaks` for the calibration.
MIN_SIGNIFICANCE = 8.0

#: Peaks closer than this are one peak. A survey cannot resolve anything
#: finer, and two tabulated lines this close are not separable on one
#: either — they are reported as an overlap instead.
MIN_DISTANCE_EV = 1.5

#: SNIP window for the survey background, in eV. Wide enough to pass under
#: the broadest Auger group and narrow enough not to follow the inelastic
#: step.
SNIP_WINDOW_EV = 30.0


@dataclass
class SurveyPeak:
    """One feature found in a survey."""

    binding_energy: float
    height: float
    significance: float
    """Prominence over the local counting noise."""
    width_ev: float
    prominence: float = 0.0

    def __str__(self) -> str:
        return f"{self.binding_energy:.1f} eV ({self.height:.0f}, {self.significance:.0f}σ)"


@dataclass
class LineMatch:
    """One tabulated feature matched to one observed peak."""

    label: str
    kind: str
    """``"fotoemisión"`` or ``"Auger"``."""
    expected_ev: float
    observed_ev: float
    height: float
    primary: bool = False

    @property
    def delta(self) -> float:
        return self.observed_ev - self.expected_ev

    def __str__(self) -> str:
        return (f"{self.label} {self.kind[0]}: {self.observed_ev:.1f} eV "
                f"({self.delta:+.1f} frente a tabla)")


@dataclass
class ElementFinding:
    """The evidence for one element being present."""

    symbol: str
    name: str
    confidence: str
    """``"alta"``, ``"media"`` or ``"baja"``."""
    matched: list[LineMatch] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    """Features that should have been visible in the measured range and
    were not. An element identified on its strongest line with its second
    line missing is worth looking at twice."""
    notes: list[str] = field(default_factory=list)

    @property
    def primary_match(self) -> Optional[LineMatch]:
        return next((m for m in self.matched if m.primary), None)

    def describe(self) -> str:
        head = f"{self.symbol} ({self.name}), confianza {self.confidence}"
        body = "; ".join(str(m) for m in self.matched)
        return f"{head}: {body}" if body else head


@dataclass
class SurveyResult:
    """What a survey scan says, and what it leaves unexplained."""

    peaks: list[SurveyPeak]
    elements: list[ElementFinding]
    uncorroborated: list[ElementFinding]
    unexplained: list[SurveyPeak]
    """Peaks no element in the library accounts for. These are the result,
    not the leftovers: a strong unexplained peak is an element that is not
    in the library, and the library here is deliberately small."""
    overlaps: list[str]
    warnings: list[str] = field(default_factory=list)
    photon_energy: Optional[float] = None

    def symbols(self) -> list[str]:
        return [item.symbol for item in self.elements]

    def summary(self) -> str:
        lines = [
            f"Survey: {len(self.peaks)} picos por encima del umbral, "
            f"{len(self.elements)} elementos identificados",
            "",
        ]
        lines.extend("  " + item.describe() for item in self.elements)
        for item in self.elements:
            for note in item.notes:
                lines.append(f"    · {note}")
        if self.uncorroborated:
            lines.append("")
            lines.append("Sin corroborar (una sola línea, que es más probable "
                         "que sea coincidencia que elemento):")
            lines.extend("  " + item.describe() for item in self.uncorroborated)
        if self.unexplained:
            lines.append("")
            lines.append("Picos sin explicar: " + ", ".join(
                str(peak) for peak in self.unexplained))
        if self.overlaps:
            lines.append("")
            lines.extend("  solape: " + text for text in self.overlaps)
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + text for text in self.warnings)
        return "\n".join(lines)


def find_survey_peaks(
    spectrum: XPSSpectrum,
    min_significance: float = MIN_SIGNIFICANCE,
    snip_window_ev: float = SNIP_WINDOW_EV,
    min_distance_ev: float = MIN_DISTANCE_EV,
) -> tuple[list[SurveyPeak], np.ndarray]:
    """Peaks in a survey, over a SNIP background.

    SNIP rather than Shirley, and deliberately: a Shirley needs endpoints
    chosen around a single region, and a survey has thirty. SNIP takes one
    parameter — a width in channels — and removes everything wider than it,
    which is exactly the question being asked of a survey background.

    The significance of a peak is its **prominence divided by the counting
    noise at that point**, and both halves of that matter. Prominence,
    because a survey's biggest features are the broad Auger groups, and
    every ripple of noise on top of one of them is a local maximum: judged
    by height alone a single Auger group yields fifty "peaks". And local
    noise, because √N at the top of a 60 000-count line is 245 and at a
    1 000-count background is 32 — a single noise level for the whole scan
    is wrong by a factor of eight from one end to the other, and it is
    wrong in the direction that invents peaks on top of the strong ones.

    The threshold is calibrated on pure Poisson noise, the same way the
    Raman and XRD ones are: over twenty synthetic 1200 eV surveys with no
    peaks in them at all, a threshold of 4 gives about 38 false peaks per
    scan, 5 gives 9, 6 gives 0.6, and 8 gives none at all. It is set at 8.

    Returns
    -------
    tuple
        ``(peaks, background)``.
    """
    counts = spectrum.counts
    step = abs(spectrum.step)
    iterations = max(4, int(round(snip_window_ev / max(step, 1e-6))))
    background = snip_baseline(counts, iterations=iterations)
    above = counts - background

    accumulated = spectrum.accumulated_counts
    if accumulated is None:
        # No way back to counts: fall back to one robust noise level for the
        # whole scan and let the caller know through the survey's warnings.
        noise = np.full_like(counts, spectrum.noise_estimate() or 1.0)
    else:
        scale = float(np.max(counts / np.clip(accumulated, 1e-9, None))) \
            if np.any(accumulated > 0) else 1.0
        noise = np.sqrt(np.clip(accumulated, 1.0, None)) * scale

    distance = max(3, int(round(min_distance_ev / max(step, 1e-6))))
    indices, properties = signal_find_peaks(above, distance=distance,
                                            prominence=0.0)
    peaks: list[SurveyPeak] = []
    for position, prominence, left, right in zip(
        indices, properties["prominences"],
        properties["left_bases"], properties["right_bases"],
    ):
        significance = float(prominence / max(noise[position], 1e-9))
        if significance < min_significance:
            continue
        centre, height = _parabolic(spectrum.binding_energy, above, int(position))
        peaks.append(
            SurveyPeak(
                binding_energy=centre,
                height=float(above[position]),
                significance=significance,
                width_ev=float(abs(spectrum.binding_energy[int(right)]
                                   - spectrum.binding_energy[int(left)])),
                prominence=float(prominence),
            )
        )
    return _merge(peaks, min_distance_ev), background


def _parabolic(x: np.ndarray, y: np.ndarray, index: int) -> tuple[float, float]:
    """Sub-channel maximum through the three points around ``index``."""
    if index <= 0 or index >= y.size - 1:       # pragma: no cover - edge channel
        return float(x[index]), float(y[index])
    y0, y1, y2 = float(y[index - 1]), float(y[index]), float(y[index + 1])
    denominator = y0 - 2.0 * y1 + y2
    if denominator == 0:                        # pragma: no cover - flat top
        return float(x[index]), y1
    offset = 0.5 * (y0 - y2) / denominator
    step = float(x[index + 1] - x[index - 1]) / 2.0
    return float(x[index] + offset * step), float(y1 - 0.25 * (y0 - y2) * offset)


def _merge(peaks: list[SurveyPeak], distance: float) -> list[SurveyPeak]:
    """Collapse peaks closer than ``distance``, keeping the taller."""
    out: list[SurveyPeak] = []
    for peak in sorted(peaks, key=lambda item: item.binding_energy):
        if out and peak.binding_energy - out[-1].binding_energy < distance:
            if peak.height > out[-1].height:
                out[-1] = peak
            continue
        out.append(peak)
    return out


def _nearest(peaks: Sequence[SurveyPeak], energy: float,
             tolerance: float) -> Optional[SurveyPeak]:
    best: Optional[SurveyPeak] = None
    for peak in peaks:
        if abs(peak.binding_energy - energy) <= tolerance:
            if best is None or abs(peak.binding_energy - energy) < \
                    abs(best.binding_energy - energy):
                best = peak
    return best


def identify(
    spectrum: XPSSpectrum,
    elements: Optional[Sequence[str]] = None,
    tolerance: float = LINE_TOLERANCE,
    min_significance: float = MIN_SIGNIFICANCE,
    database: Optional[XPSDatabase] = None,
) -> SurveyResult:
    """Identify the elements present in a survey scan.

    Three rules decide whether an element is reported as present, and all
    three exist because dropping any one of them puts elements in the
    report that are not in the sample:

    1. **Its strongest line must be there.** A match on a weak line with
       the main line absent is a coincidence: over a 1200 eV range, at
       ±2 eV, roughly one position in three is near some tabulated line.
    2. **A doublet must show the component that carries the area.** The
       2p3/2, 3d5/2 or 4f7/2 member is the one that must match; a match on
       the weak partner alone is what happens when a *different* element's
       line lands there — sulphur's 2p1/2 and selenium's 3p1/2 are 0.1 eV
       apart, and without this rule every selenide contains sulphur.
    3. **At least one matched peak must be exclusively its own.** This is
       the same rule the Raman side applies to oxides: iron's LMM Auger
       group sits on cobalt's 2p with an aluminium anode, so an
       iron-containing sample "contains cobalt" unless cobalt is required
       to show something iron cannot explain.

    Parameters
    ----------
    elements:
        Restrict the search to these symbols. ``None`` searches the whole
        library, which is what you want even when you know what you put in
        the reactor: the library's other entries are there to explain peaks
        that would otherwise be attributed to the elements you expect.
    tolerance:
        How far a peak may sit from a tabulated line, in eV.

    Returns
    -------
    SurveyResult
    """
    database = database or load_xps_database()
    if not spectrum.is_survey:
        raise XPSError(
            f"esto abarca {spectrum.range[1] - spectrum.range[0]:.0f} eV y no "
            "es un survey. La identificación por elementos necesita el "
            "barrido ancho: sobre una ventana de 20 eV, cualquier pico "
            "coincide con algo"
        )
    peaks, _ = find_survey_peaks(spectrum, min_significance)
    peaks.sort(key=lambda item: item.binding_energy)
    low, high = spectrum.range
    warnings: list[str] = []
    if spectrum.photon_energy is None:
        warnings.append(
            "sin energía del fotón no se puede situar ninguna línea Auger, y "
            "las Auger son la mitad de la corroboración de casi todos los "
            "metales. Además, un pico Auger sin identificar se atribuye a la "
            "línea de fotoemisión que le caiga cerca: declara el ánodo"
        )
    if spectrum.accumulated_counts is None:
        warnings.append(
            "el espectro está en cuentas por segundo sin tiempo por canal, "
            "así que el umbral de detección se ha aplicado con un solo nivel "
            "de ruido para todo el barrido en vez de con √N punto a punto. "
            "Sobre las líneas intensas eso es optimista"
        )

    symbols = list(elements) if elements else list(database.symbols)
    evidence: dict[str, tuple[list[LineMatch], list[str], list[SurveyPeak]]] = {}
    claims: dict[int, set[str]] = {}

    for symbol in symbols:
        element = database.element(symbol)
        matched: list[LineMatch] = []
        unmatched: list[str] = []
        hit_peaks: list[SurveyPeak] = []
        for line in element.lines:
            components = [(label, energy, share)
                          for label, energy, share in line.components()
                          if low <= energy <= high]
            if not components:
                continue
            found = {label: _nearest(peaks, energy, tolerance)
                     for label, energy, _ in components}
            strongest = max(components, key=lambda item: item[2])
            main = strongest[0]
            if found.get(main) is None:
                if line.primary:
                    unmatched.append(f"{main} ({strongest[1]:.1f} eV)")
                continue
            for label, energy, _ in components:
                peak = found[label]
                if peak is None:
                    continue
                matched.append(LineMatch(
                    label=label, kind="fotoemisión", expected_ev=energy,
                    observed_ev=peak.binding_energy, height=peak.height,
                    primary=line.primary,
                ))
                hit_peaks.append(peak)
                claims.setdefault(id(peak), set()).add(symbol)
        if spectrum.photon_energy is not None:
            for group in element.auger:
                energy = group.binding_at(spectrum.photon_energy,
                                          spectrum.work_function)
                if not low <= energy <= high:
                    continue
                peak = _nearest(peaks, energy, max(tolerance, group.width_ev / 2.0))
                if peak is None:
                    continue
                matched.append(LineMatch(
                    label=group.label, kind="Auger", expected_ev=energy,
                    observed_ev=peak.binding_energy, height=peak.height,
                ))
                hit_peaks.append(peak)
                claims.setdefault(id(peak), set()).add(symbol)
        if matched:
            evidence[symbol] = (matched, unmatched, hit_peaks)

    found_elements: list[ElementFinding] = []
    weak: list[ElementFinding] = []
    for symbol, (matched, unmatched, hit_peaks) in evidence.items():
        element = database.element(symbol)
        notes: list[str] = []
        primary = next((m for m in matched if m.primary), None)
        distinct = {id(peak) for peak in hit_peaks}
        exclusive = [peak for peak in hit_peaks
                     if claims.get(id(peak), set()) == {symbol}]
        shared_with = sorted({
            other for peak in hit_peaks
            for other in claims.get(id(peak), set()) if other != symbol
        })

        if primary is None:
            weak.append(ElementFinding(
                symbol=symbol, name=element.name, confidence="baja",
                matched=matched, unmatched=unmatched,
                notes=["la línea principal del elemento no aparece; lo que hay "
                       "es una línea débil, y a esa distancia una coincidencia "
                       "es más probable que el elemento"],
            ))
            continue
        if not exclusive:
            weak.append(ElementFinding(
                symbol=symbol, name=element.name, confidence="baja",
                matched=matched, unmatched=unmatched,
                notes=[
                    "todos sus picos los explica también "
                    + ", ".join(shared_with)
                    + ". No hay ninguna línea que sea solo suya, así que este "
                    "survey no puede decidirlo: hace falta una región de alta "
                    "resolución donde los dos no coincidan"
                ],
            ))
            continue

        in_range = _features_in_range(element, spectrum, low, high)
        if len(distinct) >= 2:
            confidence = "alta"
            notes.append("corroborado en {} picos distintos: {}".format(
                len(distinct), ", ".join(sorted({m.label for m in matched}))))
        elif in_range <= 1:
            confidence = "media"
            notes.append(
                "solo una de sus líneas cae dentro del intervalo medido, así "
                "que no hay con qué corroborarlo. Amplía el survey o mide otra "
                "región de este elemento"
            )
        else:
            confidence = "media"
            notes.append(
                f"la línea principal está, pero de las otras {in_range - 1} que "
                "deberían verse no aparece ninguna: mira si el pico es de otra "
                "cosa"
            )
        if shared_with:
            notes.append(
                "comparte picos con " + ", ".join(shared_with)
                + ", pero tiene línea propia"
            )
        found_elements.append(ElementFinding(
            symbol=symbol, name=element.name, confidence=confidence,
            matched=matched, unmatched=unmatched, notes=notes,
        ))

    overlaps = [
        f"el pico de {peak.binding_energy:.1f} eV lo reclaman "
        + " y ".join(sorted(claims[id(peak)]))
        for peak in peaks
        if len(claims.get(id(peak), set())) > 1
    ]
    unexplained = [peak for peak in peaks if not claims.get(id(peak))]
    if unexplained:
        warnings.append(
            f"{len(unexplained)} picos no los explica ningún elemento de la "
            "biblioteca, que es pequeña a propósito. El más intenso está en "
            f"{max(unexplained, key=lambda p: p.height).binding_energy:.1f} eV"
        )
    warnings.extend(_context_warnings(spectrum, found_elements, peaks))
    return SurveyResult(
        peaks=peaks, elements=found_elements, uncorroborated=weak,
        unexplained=unexplained, overlaps=overlaps, warnings=warnings,
        photon_energy=spectrum.photon_energy,
    )


def _features_in_range(element, spectrum: XPSSpectrum,
                       low: float, high: float) -> int:
    """How many of this element's features could have been seen at all."""
    total = 0
    for line in element.lines:
        total += sum(1 for _, energy, _ in line.components() if low <= energy <= high)
    if spectrum.photon_energy is not None:
        total += sum(
            1 for group in element.auger
            if low <= group.binding_at(spectrum.photon_energy,
                                       spectrum.work_function) <= high
        )
    return total


def _context_warnings(spectrum: XPSSpectrum, found: list[ElementFinding],
                      peaks: list[SurveyPeak]) -> list[str]:
    """Warnings about the measurement rather than about any one element."""
    out: list[str] = []
    if not spectrum.monochromated:
        out.append(
            "fuente no monocromada: cada línea intensa arrastra satélites de "
            "rayos X unos 8–12 eV por debajo en energía de enlace. En un "
            "survey se confunden con líneas débiles de otros elementos"
        )
    symbols = {item.symbol for item in found}
    if "C" in symbols and "O" in symbols:
        out.append(
            "carbono y oxígeno aparecen en TODA muestra expuesta al aire. Que "
            "estén no dice que sean de la muestra, y su intensidad marca "
            "cuánta contaminación hay encima — que es lo que atenúa todo lo "
            "demás"
        )
    if spectrum.step > 1.2:
        out.append(
            f"el paso del survey es {spectrum.step:.1f} eV: suficiente para "
            "identificar elementos, insuficiente para cualquier posición o "
            "anchura que se quiera interpretar"
        )
    strongest = max((peak.height for peak in peaks), default=0.0)
    if strongest > 0:
        faint = [item.symbol for item in found
                 if (item.primary_match.height if item.primary_match else 0.0)
                 < 0.01 * strongest]
        if faint:
            out.append(
                "estos elementos salen por debajo del 1 % del pico más "
                f"intenso: {', '.join(faint)}. Cerca del límite de detección "
                "de la técnica (~0.1–1 % atómico), su cuantificación tiene un "
                "error relativo grande"
            )
    return out


__all__ = [
    "ElementFinding",
    "LINE_TOLERANCE",
    "LineMatch",
    "MIN_SIGNIFICANCE",
    "SNIP_WINDOW_EV",
    "SurveyPeak",
    "SurveyResult",
    "find_survey_peaks",
    "identify",
]
