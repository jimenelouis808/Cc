"""Peak finding and phase identification in a powder pattern.

Phase identification here is a *search-match* in the classical sense, done
against calculated patterns rather than against a peak list. That
distinction is the whole reason the structures are structures: a candidate
phase can be tested at the lattice parameters your sample actually has,
not only at the ones in the card.

Three things this module does that a naive position-matcher does not.

**It fits the zero shift before judging anything.** A sample sitting
0.1 mm proud of the holder shifts every peak by a few hundredths of a
degree, systematically and towards low angle, and a matcher with a
0.05° window then rejects the correct phase. So a common offset is
estimated from the strongest peaks first and reported: if it comes out
large, that is a sample-height problem to fix at the diffractometer, not
an analysis parameter to widen.

**It judges positions and intensities separately.** Position agreement is
strong evidence, because peak positions depend only on the lattice.
Intensity agreement is weak evidence, because preferred orientation
routinely changes intensity ratios by factors of several in exactly the
layered materials this package is aimed at. Reporting them as one number
would let a texture artefact veto a correct identification, so they are
reported as two, and the ranking leans on position.

**It reports what it could not explain.** An identification that accounts
for six of nine observed peaks is a partial answer, and the three left
over are the interesting part — they are the phase you did not expect.
:attr:`PhaseSearchResult.unexplained` is not diagnostics, it is the
result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.signal import find_peaks as _scipy_find_peaks
from scipy.signal import peak_prominences, peak_widths

from ..core.baseline import asls_baseline
from ..core.compat import trapezoid
from .pattern import Pattern
from .powder import Reflection, reflections
from .structure import Crystal

#: Matching window in degrees 2θ, before the zero shift is fitted.
SEARCH_WINDOW = 0.25

#: Window used after the zero shift has been removed.
MATCH_WINDOW = 0.12

#: Calculated reflections weaker than this (relative to 100) are not
#: required to be observed: they are below a normal detection limit.
WEAK_REFLECTION = 5.0

#: Minimum matched-filter significance for a peak to count as detected.
#:
#: Calibrated, not chosen: twenty synthetic patterns of pure Poisson noise
#: on a smooth background, 3500 points each, run through this very
#: detector. The false-peak rate per pattern was 44 at a threshold of 8,
#: 3.2 at 12, 0.2 at 16 and 0 at 20. Eighteen sits just inside the clean
#: region and still finds graphite's 104 reflection, which is 0.8 % of the
#: 002. Lowering it does not find more phases; it finds more ripples on
#: the flank of the strongest peak, and reports them as unknown phases.
MIN_SIGNIFICANCE = 18.0

#: Pattern length the threshold above was calibrated on.
CALIBRATION_POINTS = 3500.0

#: A zero shift larger than this is a sample-height problem, not a fit.
LARGE_ZERO_SHIFT = 0.08


@dataclass
class XRDPeak:
    """One observed reflection, measured off the data without a model."""

    two_theta: float
    d: float
    height: float
    fwhm: Optional[float]
    area: float
    prominence: float
    significance: float

    def __str__(self) -> str:
        width = f"{self.fwhm:.3f}" if self.fwhm else "n/a"
        return (
            f"2θ={self.two_theta:8.3f}°  d={self.d:8.4f} Å  I={self.height:10.1f}  "
            f"FWHM={width}°  σ={self.significance:.0f}"
        )


def find_peaks(
    pattern: Pattern,
    min_significance: float = MIN_SIGNIFICANCE,
    baseline_lambda: float = 1e6,
    min_separation_deg: float = 0.10,
    strip_doublet: bool = True,
) -> list[XRDPeak]:
    """Locate reflections in a diffractogram.

    The background is removed with the same asymmetric-least-squares
    smoother the Raman side uses, and peaks are then thresholded on
    matched-filter significance rather than on height. The reason is the
    same as there: a broad, modest peak spanning fifty points is far more
    significant than a tall spike spanning two, and a height threshold
    finds the spike and misses the peak — which in diffraction is exactly
    backwards, since the broad one is the nanocrystalline phase you are
    looking for and the spike is a cosmic ray or a dead pixel.

    The Kα₂ satellite is stripped first when the pattern says it has one.
    Without that step every reflection is found twice above about 40° 2θ,
    the extras land in the unexplained list, and a three-phase mixture
    reports as needing a fourth phase that does not exist.
    """
    # The uncertainty is POINT-WISE, not a single number for the pattern.
    # Counting statistics make the noise on top of a 10 000-count peak a
    # hundred counts and the noise on a 100-count background ten, so one
    # global sigma either buries the weak reflections or finds dozens of
    # ripples on the flanks of the strong ones. The first version used the
    # median estimate and reported seventy "unexplained peaks" on a clean
    # single-phase pattern, every one of them a ripple beside the 002.
    noise = np.maximum(
        np.asarray(pattern.sigma, dtype=float)
        if pattern.sigma is not None
        else np.full(pattern.n, pattern.noise_estimate()),
        1e-9,
    )
    if not pattern.counts:
        noise = np.full(pattern.n, max(pattern.noise_estimate(), 1e-9))
    if strip_doublet and pattern.has_doublet:
        from .preprocess import strip_kalpha2

        pattern = strip_kalpha2(pattern)
    y = np.asarray(pattern.intensity, dtype=float)
    background = asls_baseline(y, lam=baseline_lambda, p=0.001)
    corrected = y - background
    step = max(pattern.step, 1e-6)
    distance = max(1, int(round(min_separation_deg / step)))

    threshold = min_significance * _trials_factor(pattern.n)
    indices, _ = _scipy_find_peaks(corrected, distance=distance)
    if indices.size == 0:
        return []
    prominences = peak_prominences(corrected, indices)[0]
    widths, _, left, right = peak_widths(corrected, indices, rel_height=0.5)

    peaks: list[XRDPeak] = []
    for position, prominence, width, a, b in zip(
        indices, prominences, widths, left, right
    ):
        height = float(corrected[position])
        if height <= 0.0:
            continue
        fwhm = float(width * step) if width > 0 else None
        span = max(width, 1.0)
        significance = height * math.sqrt(span) / float(noise[position])
        if significance < threshold:
            continue
        low, high = int(math.floor(a)), int(math.ceil(b)) + 1
        area = float(trapezoid(corrected[low:high], pattern.two_theta[low:high]))
        angle = _refine(pattern.two_theta, corrected, int(position))
        d = pattern.wavelength / (2.0 * math.sin(math.radians(angle) / 2.0))
        peaks.append(
            XRDPeak(
                two_theta=angle,
                d=d,
                height=height,
                fwhm=fwhm,
                area=area,
                prominence=float(prominence),
                significance=float(significance),
            )
        )
    peaks.sort(key=lambda p: p.two_theta)
    return peaks


def _trials_factor(n_points: int) -> float:
    """Look-elsewhere scaling of the detection threshold.

    The expected maximum of *n* independent samples grows as
    ``sqrt(2 ln n)``, so holding the false-alarm rate fixed needs a
    threshold growing as ``sqrt(ln n)``. Normalised to the 3500-point
    pattern the default was calibrated on, so that a 20–30° quick scan is
    not held to the standard of a 10–120° overnight one.
    """
    if n_points <= 1:
        return 1.0
    return float(math.sqrt(math.log(n_points) / math.log(CALIBRATION_POINTS)))


def _refine(x: np.ndarray, y: np.ndarray, index: int) -> float:
    """Parabolic interpolation of a maximum, for sub-step positions."""
    if index <= 0 or index >= len(x) - 1:
        return float(x[index])
    y0, y1, y2 = float(y[index - 1]), float(y[index]), float(y[index + 1])
    denominator = y0 - 2.0 * y1 + y2
    if abs(denominator) < 1e-12:
        return float(x[index])
    offset = 0.5 * (y0 - y2) / denominator
    offset = max(-1.0, min(1.0, offset))
    return float(x[index] + offset * (x[index + 1] - x[index]))


# -- matching ---------------------------------------------------------


@dataclass
class PhaseMatch:
    """How well one candidate structure explains an observed pattern."""

    crystal: Crystal
    matched: list[tuple[Reflection, XRDPeak]] = field(default_factory=list)
    missing: list[Reflection] = field(default_factory=list)
    """Strong calculated reflections with no observed peak. The damning
    evidence against a phase: an extra peak can belong to something else,
    a missing strong reflection cannot be explained away."""
    zero_shift: float = 0.0
    expected_strong: int = 0

    @property
    def coverage(self) -> float:
        """Fraction of the phase's strong reflections that were found."""
        if not self.expected_strong:
            return 0.0
        found = sum(1 for reflection, _ in self.matched
                    if reflection.intensity >= WEAK_REFLECTION)
        return min(1.0, found / self.expected_strong)

    @property
    def position_error(self) -> float:
        """RMS position mismatch in degrees, after the zero shift."""
        if not self.matched:
            return float("inf")
        errors = [
            peak.two_theta - (reflection.two_theta + self.zero_shift)
            for reflection, peak in self.matched
        ]
        return float(np.sqrt(np.mean(np.square(errors))))

    @property
    def intensity_agreement(self) -> Optional[float]:
        """Spearman correlation of calculated and observed intensities.

        Rank correlation rather than Pearson because preferred orientation
        scales whole families of reflections: it wrecks the linear
        relationship while usually preserving the order. ``None`` when
        fewer than four reflections matched, where a correlation
        coefficient means nothing.
        """
        if len(self.matched) < 4:
            return None
        calculated = np.array([r.intensity for r, _ in self.matched])
        observed = np.array([p.area if p.area > 0 else p.height for _, p in self.matched])
        return float(_spearman(calculated, observed))

    @property
    def score(self) -> float:
        """Overall figure of merit, 0–1, dominated by position agreement.

        ``coverage`` says how much of the phase is there and
        ``position_error`` how well it fits; intensity agreement enters
        with a small weight and can never sink an otherwise good match,
        because texture routinely wrecks intensities in exactly the
        layered materials this package targets.
        """
        if not self.matched:
            return 0.0
        precision = math.exp(-(self.position_error / MATCH_WINDOW) ** 2)
        agreement = self.intensity_agreement
        bonus = 0.0 if agreement is None else 0.15 * max(0.0, agreement)
        return float(min(1.0, 0.85 * self.coverage * precision + bonus))

    @property
    def verdict(self) -> str:
        score = self.score
        if score >= 0.75:
            return "presente"
        if score >= 0.45:
            return "probable"
        if score >= 0.2:
            return "dudosa"
        return "descartada"

    def summary(self) -> str:
        agreement = self.intensity_agreement
        lines = [
            f"{self.crystal.name} [{self.crystal.formula}] — {self.verdict} "
            f"(FOM {self.score:.2f})",
            f"    reflexiones fuertes halladas: "
            f"{sum(1 for r, _ in self.matched if r.intensity >= WEAK_REFLECTION)}"
            f"/{self.expected_strong}"
            f"   error de posición {self.position_error:.3f}°",
        ]
        if agreement is not None:
            lines.append(
                f"    acuerdo de intensidades (rango): {agreement:+.2f}"
                + (
                    "  — bajo; en un material laminar eso suele ser textura, "
                    "no una fase equivocada"
                    if agreement < 0.4
                    else ""
                )
            )
        if self.missing:
            worst = sorted(self.missing, key=lambda r: -r.intensity)[:3]
            lines.append(
                "    faltan reflexiones fuertes: "
                + ", ".join(
                    f"{r.hkl} a {r.two_theta + self.zero_shift:.2f}° (I={r.intensity:.0f})"
                    for r in worst
                )
            )
        return "\n".join(lines)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rho, without pulling in scipy.stats."""
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks

    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = math.sqrt(float((ra**2).sum() * (rb**2).sum()))
    if denominator <= 0.0:
        return 0.0
    return float((ra * rb).sum() / denominator)


def fit_zero_shift(
    peaks: Sequence[XRDPeak],
    calculated: Sequence[Reflection],
    window: float = SEARCH_WINDOW,
) -> float:
    """Estimate a common 2θ offset from the strongest matched pairs.

    The median of the individual offsets, not the mean: one mismatched
    pair would drag a mean across the whole window, and a robust estimate
    of a systematic shift is exactly what a median is for.
    """
    offsets: list[float] = []
    strong = sorted(calculated, key=lambda r: -r.intensity)[:8]
    for reflection in strong:
        nearby = [p for p in peaks if abs(p.two_theta - reflection.two_theta) <= window]
        if not nearby:
            continue
        closest = min(nearby, key=lambda p: abs(p.two_theta - reflection.two_theta))
        offsets.append(closest.two_theta - reflection.two_theta)
    if len(offsets) < 2:
        return 0.0
    return float(np.median(offsets))


def match_phase(
    peaks: Sequence[XRDPeak],
    crystal: Crystal,
    pattern: Pattern,
    fit_zero: bool = True,
    window: float = MATCH_WINDOW,
) -> PhaseMatch:
    """Test one candidate structure against an observed peak list."""
    low, high = pattern.range
    calculated = reflections(
        crystal, wavelength=pattern.wavelength, two_theta_range=(low, high)
    )
    match = PhaseMatch(crystal=crystal)
    if not calculated:
        return match
    match.zero_shift = fit_zero_shift(peaks, calculated) if fit_zero else 0.0
    match.expected_strong = sum(1 for r in calculated if r.intensity >= WEAK_REFLECTION)

    used: set[int] = set()
    for reflection in sorted(calculated, key=lambda r: -r.intensity):
        target = reflection.two_theta + match.zero_shift
        best, best_distance = None, window
        for index, peak in enumerate(peaks):
            if index in used:
                continue
            distance = abs(peak.two_theta - target)
            if distance <= best_distance:
                best, best_distance = index, distance
        if best is None:
            if reflection.intensity >= WEAK_REFLECTION:
                match.missing.append(reflection)
            continue
        used.add(best)
        match.matched.append((reflection, peaks[best]))
    match.matched.sort(key=lambda pair: pair[1].two_theta)
    return match


@dataclass
class PhaseSearchResult:
    """The outcome of a search-match over a library of candidates."""

    accepted: list[PhaseMatch] = field(default_factory=list)
    rejected: list[PhaseMatch] = field(default_factory=list)
    unexplained: list[XRDPeak] = field(default_factory=list)
    kalpha2_residuals: list[XRDPeak] = field(default_factory=list)
    """Leftovers sitting exactly one Kα₂ separation above an explained
    peak. Rachinger stripping removes about 90 % of the satellite, and on
    a strong reflection the last 10 % is still a detectable maximum. They
    are set aside rather than reported as unknown phases, because calling
    the tail of a copper Kα₂ line an unidentified phase is worse than
    saying nothing."""
    peaks: list[XRDPeak] = field(default_factory=list)
    zero_shift: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def phases(self) -> list[Crystal]:
        return [m.crystal for m in self.accepted]

    def summary(self) -> str:
        lines: list[str] = []
        if not self.peaks:
            return "No se ha detectado ningún pico en el difractograma."
        lines.append(f"Picos detectados: {len(self.peaks)}")
        if abs(self.zero_shift) > 1e-6:
            lines.append(
                f"Desplazamiento de cero ajustado: {self.zero_shift:+.4f}° 2θ"
            )
        lines.append("")
        if self.accepted:
            lines.append("Fases identificadas:")
            lines.extend("  " + m.summary() for m in self.accepted)
        else:
            lines.append("Ninguna fase de la biblioteca explica el difractograma.")
        if self.rejected:
            lines.append("")
            lines.append("Descartadas (mejores tres):")
            for match in self.rejected[:3]:
                lines.append(
                    f"  {match.crystal.name:22s} FOM {match.score:.2f} — "
                    f"{len(match.missing)} reflexión(es) fuerte(s) sin aparecer"
                )
        if self.kalpha2_residuals:
            lines.append("")
            lines.append(
                f"{len(self.kalpha2_residuals)} máximo(s) descartado(s) por ser "
                "cola de Kα₂ sobre un pico ya explicado (el pelado de "
                "Rachinger deja en torno al 10 %). No son fases"
            )
        if self.unexplained:
            lines.append("")
            lines.append(
                f"Picos SIN explicar ({len(self.unexplained)}). Esto no es un "
                "resto: es el resultado más informativo del análisis, porque "
                "es la fase que no esperabas:"
            )
            for peak in sorted(self.unexplained, key=lambda p: -p.height)[:8]:
                lines.append("    " + str(peak))
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


def identify_phases(
    pattern: Pattern,
    candidates: Optional[Sequence[Crystal]] = None,
    max_phases: int = 4,
    accept_score: float = 0.45,
    extra_directories: Optional[Sequence[str]] = None,
    peaks: Optional[Sequence[XRDPeak]] = None,
) -> PhaseSearchResult:
    """Identify the phases present, greedily and with a residual.

    Phases are accepted one at a time, best first, and after each one the
    peaks it explains are removed before the next is judged. That is what
    stops a second phase scoring well purely on the first phase's peaks —
    the failure mode of ranking every candidate independently against the
    full pattern, which reliably "finds" three iron oxides in a sample
    containing one.

    Parameters
    ----------
    pattern:
        The measured diffractogram.
    candidates:
        Structures to test. Defaults to the whole reference library plus
        anything in ``extra_directories``.
    max_phases:
        Stop after this many. Four is already generous for a real sample;
        a fit that needs eight is usually a wrong wavelength or an
        uncorrected zero shift, not eight phases.
    accept_score:
        Figure of merit above which a phase is reported as present.
    peaks:
        Pre-computed peak list, to avoid finding them twice.

    Returns
    -------
    PhaseSearchResult
    """
    if candidates is None:
        from .reference import library_crystals

        candidates = library_crystals(extra_directories)

    observed = list(peaks) if peaks is not None else find_peaks(pattern)
    result = PhaseSearchResult(peaks=observed)
    if not observed:
        result.warnings.append(
            "no se ha detectado ningún pico por encima del umbral de "
            "significancia. O el difractograma es de material amorfo, o el "
            "fondo se ha comido las reflexiones, o el rango medido no "
            "contiene ninguna"
        )
        return result

    remaining = list(observed)
    all_matches: list[PhaseMatch] = []
    for _ in range(max_phases):
        scored = [match_phase(remaining, c, pattern) for c in candidates
                  if c.name not in {m.crystal.name for m in result.accepted}]
        scored = [m for m in scored if m.matched]
        if not scored:
            break
        best = max(scored, key=lambda m: m.score)
        all_matches.extend(scored)
        if best.score < accept_score:
            break
        result.accepted.append(best)
        claimed = {id(peak) for _, peak in best.matched}
        remaining = [p for p in remaining if id(p) not in claimed]
        if not remaining:
            break

    result.unexplained, result.kalpha2_residuals = _split_kalpha2_residuals(
        remaining, result.accepted, pattern
    )
    seen = {m.crystal.name for m in result.accepted}
    result.rejected = sorted(
        (m for m in all_matches if m.crystal.name not in seen),
        key=lambda m: -m.score,
    )
    if result.accepted:
        result.zero_shift = float(np.median([m.zero_shift for m in result.accepted]))

    _add_warnings(result, pattern)
    return result


def _split_kalpha2_residuals(
    remaining: Sequence[XRDPeak],
    accepted: Sequence[PhaseMatch],
    pattern: Pattern,
    tolerance: float = 0.04,
) -> tuple[list[XRDPeak], list[XRDPeak]]:
    """Separate genuine leftovers from Kα₂ tails.

    Rachinger stripping takes out roughly 90 % of the satellite; on a
    strong reflection the remaining tenth is still a local maximum well
    above the noise. Those sit at a known place — one Kα₂ separation above
    an explained peak — so they can be identified and set aside instead of
    being reported as an unknown phase, which is the worse error.
    """
    if not pattern.has_doublet or not accepted:
        return list(remaining), []
    from .preprocess import kalpha2_offset

    explained = np.array(
        sorted(peak.two_theta for match in accepted for _, peak in match.matched)
    )
    if explained.size == 0:
        return list(remaining), []
    offsets = kalpha2_offset(explained, pattern.wavelength, pattern.kalpha2_lambda)
    satellites = explained + offsets

    genuine: list[XRDPeak] = []
    residuals: list[XRDPeak] = []
    for peak in remaining:
        if np.min(np.abs(satellites - peak.two_theta)) <= tolerance:
            residuals.append(peak)
        else:
            genuine.append(peak)
    return genuine, residuals


def _add_warnings(result: PhaseSearchResult, pattern: Pattern) -> None:
    """Say what a result implies about the measurement, not just the sample."""
    if abs(result.zero_shift) > LARGE_ZERO_SHIFT:
        result.warnings.append(
            f"el desplazamiento de cero ajustado es {result.zero_shift:+.3f}°, "
            "grande. Casi siempre significa que la muestra no está a la altura "
            "del plano de referencia del portamuestras, no que el equipo esté "
            "descalibrado. Corrígelo en el difractómetro: un desplazamiento de "
            "muestra deforma las posiciones de forma NO lineal en 2θ y ningún "
            "cero constante lo arregla del todo. Verifícalo con silicio"
        )
    unexplained_strong = [
        p for p in result.unexplained
        if result.peaks and p.height >= 0.1 * max(q.height for q in result.peaks)
    ]
    if unexplained_strong:
        result.warnings.append(
            f"quedan {len(unexplained_strong)} pico(s) intensos sin explicar. "
            "La biblioteca de referencia que has usado no contiene esa fase: "
            "descarga su CIF de la Crystallography Open Database y añádelo con "
            "--cif. Un ajuste de Rietveld sin ella repartirá su intensidad "
            "entre las fases que sí están y sesgará todas las fracciones"
        )
    from .scattering import fluorescence_risk

    elements = {
        site.element for match in result.accepted for site in match.crystal.sites
    }
    if elements:
        risk = fluorescence_risk(sorted(elements), pattern.anode)
        if risk:
            result.warnings.append(risk)
    if result.accepted and not pattern.covers(*pattern.range, fraction=1.0):
        pass


__all__ = [
    "LARGE_ZERO_SHIFT",
    "MATCH_WINDOW",
    "MIN_SIGNIFICANCE",
    "SEARCH_WINDOW",
    "WEAK_REFLECTION",
    "PhaseMatch",
    "PhaseSearchResult",
    "XRDPeak",
    "find_peaks",
    "fit_zero_shift",
    "identify_phases",
    "match_phase",
]
