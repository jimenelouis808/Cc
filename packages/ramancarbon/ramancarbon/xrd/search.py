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

#: Fraction of a peak's own FWHM added to the matching window.
#:
#: A fixed window assumes every peak's position is known to the same
#: precision, which is true of a well-crystallised powder and false of
#: everything this package is aimed at. A CVD sample's graphite 002 is
#: three to five degrees wide and its centroid sits wherever the
#: interlayer spacing put it -- 3.44 A rather than graphite's 3.354 gives
#: 25.9 degrees instead of 26.5, a shift of 0.6 that a 0.12 window cannot
#: reach. So every reflection under a broad peak read as unexplained, and
#: nothing was ever identified.
#:
#: Half the FWHM is the honest tolerance: a reflection anywhere under the
#: peak is a plausible assignment, and demanding better than the peak's
#: own width is demanding a precision the measurement does not have. The
#: cost is paid in the score rather than hidden -- see
#: :meth:`PhaseMatch.score`, where the position error is judged against
#: the window that was actually used.
WIDTH_TOLERANCE = 0.5

#: How far above the detection threshold a missing reflection has to be
#: predicted before its absence counts against the phase.
#:
#: At exactly the threshold a reflection is found about half the time, so
#: "it is not there" carries no information; and the prediction is a
#: known over-estimate, because it scales a clean measured height by a
#: calculated intensity and takes no account of what the detector loses
#: to a neighbour raising the local minima or to smoothing at a scale
#: that does not match the reflection's own width. Measured on the CVD
#: test pattern: the turbostratic 004 is predicted at 23 and observed at
#: 14, and the carbon was rejected for the absence of a reflection the
#: same code could not have found. Two is that factor rounded up. It
#: cannot be used to admit a phase on its own — ``intensity_coverage``
#: still requires that most of the phase's calculated intensity was
#: actually seen.
DETECTION_MARGIN = 2.0

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

#: Widths, in degrees 2θ, at which the broad pass looks. A reflection
#: wider than the first of these is one the narrow search cannot measure.
BROAD_SCALES_DEG: tuple[float, ...] = (0.6, 1.2, 2.5, 5.0)

#: How many times the broadest reportable reflection the background's
#: cutoff period is set to, by the same argument as
#: :data:`ramancarbon.core.baseline.CUTOFF_BAND_WIDTHS` on the Raman side:
#: a baseline whose cutoff sits inside a reflection follows it and takes
#: its intensity away. Three was chosen by measurement rather than by
#: analogy — see :func:`baseline_lambda_for`.
CUTOFF_REFLECTION_WIDTHS = 3.0

#: Bounds on the derived stiffness. The floor is the fixed value this
#: module used before the rule existed, so no pattern gets a SOFTER
#: background than it used to; the ceiling is where the banded solve
#: starts to lose conditioning.
BASELINE_LAMBDA_LIMITS = (1e6, 1e12)


def baseline_lambda_for(pattern: Pattern) -> float:
    """Background stiffness matched to the pattern's sampling step.

    A fixed ``lam`` is a fixed cutoff **in points**, and a point is worth
    a different number of degrees on every diffractometer. The value this
    module used before, 1e6, puts the half-power cutoff at 199 points,
    which is 3.97° on a 0.02° step — *narrower* than the broadest
    reflection the search is willing to report. The consequence is not
    subtle and it is silent: on a simulated CVD pattern the asymmetric
    least-squares background followed the turbostratic 002 and took a
    quarter of its height with it, the reflection's significance fell
    from 22 to 16 against a threshold of 18, and the pattern identified
    nothing at all while the hump stayed plainly visible on screen.

    So the cutoff is set from the sample instead: three times the
    broadest reflection the broad pass can report. That is the same
    reasoning as :func:`ramancarbon.core.baseline.auto_lambda`, with the
    factor lowered from five because a diffraction background has real
    structure — fluorescence, air scatter, the sample holder — over the
    tens of degrees a factor of five would protect.
    """
    from ..core.baseline import lambda_for_cutoff

    period = CUTOFF_REFLECTION_WIDTHS * BROAD_SCALES_DEG[-1] / max(pattern.step, 1e-6)
    return float(np.clip(lambda_for_cutoff(period), *BASELINE_LAMBDA_LIMITS))


def _window_noise(noise: np.ndarray, width_points: float) -> np.ndarray:
    """Per-point noise level averaged over a feature's own width.

    The significance of a broad feature is its mean amplitude times the
    square root of its span, over the per-point noise level *in that
    neighbourhood*. Reading that level off the single point at the
    maximum is a lottery when the counts are low: with a background of
    three counts, √N of one sample is 1, 2 or 3 depending on which
    Poisson draw landed there, so the same reflection scores 12 or 25
    according to luck. Averaging the variance over the feature's own
    width keeps the point-wise philosophy — the noise on a 10 000-count
    peak is still a hundred and on a 100-count background still ten —
    while estimating it from the hundred points the statistic actually
    used.
    """
    from scipy.ndimage import uniform_filter1d

    span = max(3, int(round(width_points)) | 1)
    return np.sqrt(
        np.maximum(uniform_filter1d(noise**2, span, mode="nearest"), 1e-18)
    )


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
    baseline_lambda: Optional[float] = None,
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
    if baseline_lambda is None:
        baseline_lambda = baseline_lambda_for(pattern)
    # The broad pass gets the pattern as measured. Rachinger stripping
    # corrects a splitting of a few hundredths of a degree, which cannot
    # matter to a reflection several degrees wide, and it is not free:
    # the recursion subtracts a shifted copy of the data, so on a
    # photon-starved pattern it costs the broad 002 a fifth of its
    # prominence and correlates its neighbours. The narrow pass, where
    # the doublet is the difference between finding one reflection and
    # two, still gets the stripped one.
    raw = np.asarray(pattern.intensity, dtype=float)
    broad_corrected = raw - asls_baseline(raw, lam=baseline_lambda, p=0.001)
    if strip_doublet and pattern.has_doublet:
        from .preprocess import strip_kalpha2

        pattern = strip_kalpha2(pattern)
        y = np.asarray(pattern.intensity, dtype=float)
        corrected = y - asls_baseline(y, lam=baseline_lambda, p=0.001)
    else:
        corrected = broad_corrected
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
        # PROMINENCE, not height. The distinction is invisible on a clean
        # pattern with a flat background and decisive on a nanocrystalline
        # one: a noise ripple riding on a 4-degree-wide graphite 002 sits
        # at the hump's own intensity, so scoring it by height hands it
        # the hump's significance. On a simulated CVD pattern that put
        # thirteen ripples of 0.02-0.09 degrees FWHM into the peak list,
        # every one of them "significant", every one unexplained -- which
        # is exactly what a real CVD pattern was doing.
        significance = float(prominence) * math.sqrt(span) / float(noise[position])
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
    peaks.extend(_broad_pass(pattern, broad_corrected, noise, threshold, peaks))
    peaks.sort(key=lambda p: p.two_theta)
    return peaks


def strongest_below_threshold(
    pattern: Pattern,
    min_significance: float = MIN_SIGNIFICANCE,
    baseline_lambda: Optional[float] = None,
) -> Optional[tuple[float, float, float]]:
    """The best maximum that did NOT clear the detection threshold.

    Silence is the least useful answer a peak search can give, and on a
    photon-starved pattern it is the usual one. The threshold is
    calibrated against pure noise and lowering it quietly would trade a
    real false-alarm rate for the appearance of working; saying how close
    the best candidate came costs nothing and hands the decision back,
    with a number attached.

    Returns ``(2θ, width in degrees, significance)`` or ``None``.
    """
    from scipy.ndimage import gaussian_filter1d

    noise = np.maximum(
        np.asarray(pattern.sigma, dtype=float)
        if pattern.sigma is not None
        else np.full(pattern.n, pattern.noise_estimate()),
        1e-9,
    )
    if not pattern.counts:
        noise = np.full(pattern.n, max(pattern.noise_estimate(), 1e-9))
    if baseline_lambda is None:
        baseline_lambda = baseline_lambda_for(pattern)
    y = np.asarray(pattern.intensity, dtype=float)
    corrected = y - asls_baseline(y, lam=baseline_lambda, p=0.001)
    step = max(pattern.step, 1e-6)
    threshold = min_significance * _trials_factor(pattern.n)

    best: Optional[tuple[float, float, float]] = None
    for width_deg in (0.1, 0.2, 0.4) + BROAD_SCALES_DEG:
        width_points = width_deg / step
        if width_points < 3.0 or width_points > pattern.n / 4:
            continue
        smoothed = gaussian_filter1d(corrected, width_points / 2.3548,
                                     mode="nearest")
        indices, _ = _scipy_find_peaks(smoothed, distance=max(1, int(width_points)))
        if indices.size == 0:
            continue
        prominences = peak_prominences(smoothed, indices)[0]
        local = _window_noise(noise, width_points)
        significance = prominences * math.sqrt(width_points) / local[indices]
        for position, value in zip(indices, significance):
            if value >= threshold:
                continue
            if best is None or value > best[2]:
                best = (float(pattern.two_theta[position]), float(width_deg),
                        float(value))
    return best


def _broad_pass(
    pattern: Pattern,
    corrected: np.ndarray,
    noise: np.ndarray,
    threshold: float,
    found: Sequence[XRDPeak],
) -> list[XRDPeak]:
    """A second look, for reflections too broad for the first one to see.

    The search above measures a peak's width with ``peak_widths``, which
    walks outward from the top until the signal falls to half. On a noisy
    pattern that walk stops at the first noise dip, so a graphite 002 four
    and a half degrees wide is measured as a twentieth of a degree wide,
    its matched-filter significance collapses with the square root of that
    width, and it is not detected at all — on a pattern where the hump is
    plainly visible by eye. That is not a threshold that needs lowering:
    it is a width that was never measured.

    So a broad feature is looked for at its own scale. The corrected
    pattern is smoothed to each of :data:`BROAD_SCALES_DEG` in turn, which
    removes the dips that stopped the walk, and any maximum that clears
    the same threshold at that scale is kept with that scale as its width.

    It is deliberately ADDITIVE and deliberately second. Anything the
    narrow search already found keeps the width the narrow search
    measured, because for a resolved reflection that measurement is the
    better one; a broad candidate within half its width of an existing
    peak is dropped rather than competing with it. What this pass can
    contribute is only what the first one structurally cannot see, and on
    every pattern whose peaks are resolved it contributes nothing.
    """
    from scipy.ndimage import gaussian_filter1d

    step = max(pattern.step, 1e-6)
    # Each claimed position carries its own width: a broad candidate half a
    # degree from a peak already measured at three and a half degrees wide
    # is the same peak, and judging the gap against only the candidate's
    # width let it through as a second one — which then let a second
    # turbostratic carbon be "identified" on it.
    claimed = [(peak.two_theta, peak.fwhm or 0.0) for peak in found]
    extra: list[XRDPeak] = []
    for width_deg in BROAD_SCALES_DEG:
        width_points = width_deg / step
        if width_points < 3.0 or width_points > pattern.n / 4:
            continue
        smoothed = gaussian_filter1d(corrected, width_points / 2.3548,
                                     mode="nearest")
        indices, _ = _scipy_find_peaks(smoothed, distance=max(1, int(width_points)))
        if indices.size == 0:
            continue
        prominences = peak_prominences(smoothed, indices)[0]
        local = _window_noise(noise, width_points)
        for position, prominence in zip(indices, prominences):
            significance = prominence * math.sqrt(width_points) / local[position]
            if significance < threshold:
                continue
            angle = float(pattern.two_theta[position])
            if any(abs(angle - other) < max(0.5 * width_deg, 0.5 * other_width)
                   for other, other_width in claimed):
                continue
            claimed.append((angle, width_deg))
            span = max(1, int(round(width_points)))
            low = max(0, int(position) - span)
            high = min(len(corrected), int(position) + span + 1)
            area = float(trapezoid(corrected[low:high],
                                   pattern.two_theta[low:high]))
            height = float(smoothed[position])
            d = pattern.wavelength / (2.0 * math.sin(math.radians(angle) / 2.0))
            extra.append(
                XRDPeak(
                    two_theta=angle,
                    d=d,
                    height=height,
                    fwhm=float(width_deg),
                    area=area,
                    prominence=float(prominence),
                    significance=float(significance),
                )
            )
    return extra


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
    undetectable: list[Reflection] = field(default_factory=list)
    """Calculated reflections strong enough to be listed but too weak, at
    this phase's fitted scale and this pattern's noise, for the peak
    finder to have found them. Their absence is not evidence: see
    :meth:`coverage`."""
    zero_shift: float = 0.0
    expected_strong: int = 0
    window_used: float = MATCH_WINDOW
    """Mean matching window over the matched reflections, in degrees. Wider
    than :data:`MATCH_WINDOW` when the peaks are broad; the score is judged
    against this rather than against the constant."""
    _calculated_total: float = 0.0
    """Summed calculated intensity of every reflection inside the measured
    range. The denominator of :attr:`intensity_coverage`."""

    @property
    def coverage(self) -> float:
        """Fraction of the phase's *detectable* strong reflections found.

        Detectable is the operative word, and leaving it out is why a
        nanocrystalline carbon never identified. A turbostratic carbon's
        pattern is one enormous 002 and a handful of reflections at 5-6 %
        of it; spread over three or four degrees, a 6 % reflection is
        below the noise, so it is never found, so coverage came out at
        0.5 with a perfect position match and the phase scored under the
        acceptance threshold. The absence of a reflection the measurement
        could not have shown is not evidence against the phase — the same
        rule the Raman side applies to a band outside the measured range.
        """
        expected = self.expected_strong - len(self.undetectable)
        if expected <= 0:
            # Every strong reflection is below the noise. One matched peak
            # is then all the evidence there can be, and it is thin: the
            # score's coverage term says so rather than dividing by zero.
            return 1.0 if self.matched else 0.0
        found = sum(1 for reflection, _ in self.matched
                    if reflection.intensity >= WEAK_REFLECTION)
        return min(1.0, found / expected)

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
    def intensity_coverage(self) -> float:
        """Share of the phase's calculated intensity that was actually seen.

        The guard on the "too weak to detect" excuse, and it is needed:
        without it a phase that matched ONE weak peak declared all its
        other reflections undetectable — because the scale fitted to that
        one peak makes them so — and scored a perfect coverage. On a
        simulated CVD pattern that accepted MoSe₂, on a single line, in a
        sample containing no molybdenum.

        The difference between that and a genuine one-line phase is
        quantitative and clean. A turbostratic carbon really is 88 % of
        its diffracted intensity in the 002 alone; MoSe₂'s strongest
        reflection is 30 % of its total. So the question is not how many
        reflections were found but how much of the phase they account
        for, and that is what this measures.
        """
        if not self.matched or not self._calculated_total:
            return 0.0
        seen = sum(reflection.intensity for reflection, _ in self.matched)
        return float(min(1.0, seen / self._calculated_total))

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
        # Against the window actually used, not the constant. Judging a
        # 4-degree-wide peak matched to within 0.6 degrees by a 0.12
        # yardstick scores it zero, which would undo the whole point of
        # allowing the wider window in the first place.
        precision = math.exp(-(self.position_error / max(self.window_used, 1e-6)) ** 2)
        agreement = self.intensity_agreement
        bonus = 0.0 if agreement is None else 0.15 * max(0.0, agreement)
        seen = self.intensity_coverage
        return float(min(1.0, 0.85 * self.coverage * precision * seen + bonus))

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
    match._calculated_total = float(sum(r.intensity for r in calculated))

    used: set[int] = set()
    windows: list[float] = []
    for reflection in sorted(calculated, key=lambda r: -r.intensity):
        target = reflection.two_theta + match.zero_shift
        best, best_distance, best_window = None, None, window
        for index, peak in enumerate(peaks):
            if index in used:
                continue
            # Each peak brings its own tolerance. A sharp reflection is
            # still held to 0.12 degrees; a 4-degree-wide nanocrystalline
            # hump is not, because nothing about that measurement locates
            # a reflection to a tenth of a degree.
            allowed = window + WIDTH_TOLERANCE * float(peak.fwhm or 0.0)
            distance = abs(peak.two_theta - target)
            if distance <= allowed and (best_distance is None
                                        or distance < best_distance):
                best, best_distance, best_window = index, distance, allowed
        if best is None:
            if reflection.intensity >= WEAK_REFLECTION:
                match.missing.append(reflection)
            continue
        used.add(best)
        windows.append(best_window)
        match.matched.append((reflection, peaks[best]))
    match.matched.sort(key=lambda pair: pair[1].two_theta)
    match.window_used = (float(np.mean(windows)) if windows else window)
    _mark_undetectable(match, pattern)
    return match


def _mark_undetectable(match: PhaseMatch, pattern: Pattern) -> None:
    """Move missing reflections the measurement could not have shown.

    The phase's scale comes from the reflections that DID match — median
    of observed height over calculated intensity — and every missing one
    is then predicted at that scale. A prediction below the detection
    threshold means the peak finder was never going to find it, whatever
    the phase's abundance, so its absence says nothing about whether the
    phase is there.

    Width matters as much as height: the same integrated intensity spread
    over four degrees instead of a tenth of one is thirty times shorter,
    which is precisely the situation in a nanocrystalline sample and
    precisely where the naive rule went wrong.

    The comparison is against :data:`DETECTION_MARGIN` times the
    threshold, not the threshold itself, because a reflection predicted
    at exactly the threshold is found about half the time and its absence
    decides nothing.
    """
    if not match.matched or not match.missing:
        return
    scales = [peak.height / reflection.intensity
              for reflection, peak in match.matched
              if reflection.intensity > 1e-9 and peak.height > 0.0]
    if not scales:
        return
    scale = float(np.median(scales))
    widths = [peak.fwhm for _, peak in match.matched if peak.fwhm]
    width = float(np.median(widths)) if widths else MATCH_WINDOW
    step = max(pattern.step, 1e-9)
    noise = max(pattern.noise_estimate(), 1e-9)
    threshold = MIN_SIGNIFICANCE * _trials_factor(pattern.n)

    still_missing: list[Reflection] = []
    for reflection in match.missing:
        height = scale * reflection.intensity
        span = max(width / step, 1.0)
        if height * math.sqrt(span) / noise < threshold * DETECTION_MARGIN:
            match.undetectable.append(reflection)
        else:
            still_missing.append(reflection)
    match.missing = still_missing


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
    note = pattern.metadata.get("sigma_note")
    if note:
        result.warnings.append(note)
    if not observed:
        message = (
            "no se ha detectado ningún pico por encima del umbral de "
            "significancia. O el difractograma es de material amorfo, o el "
            "fondo se ha comido las reflexiones, o el rango medido no "
            "contiene ninguna"
        )
        near = strongest_below_threshold(pattern)
        if near is not None:
            angle, width, value = near
            threshold = MIN_SIGNIFICANCE * _trials_factor(pattern.n)
            message += (
                f".   Lo que MÁS cerca estuvo: un máximo en 2θ = {angle:.2f}°, "
                f"de unos {width:g}° de ancho, con significancia {value:.1f} "
                f"frente al umbral de {threshold:.0f}. Si lo ves en la gráfica "
                "y te parece real, baja el umbral en «Búsqueda de picos» hasta "
                f"{max(5.0, value - 1):.0f} y vuelve a identificar — pero mira "
                "primero si es un pico o es el fondo, porque el umbral está "
                "calibrado contra ruido puro y por debajo de él el programa "
                "empieza a inventarse máximos"
            )
        result.warnings.append(message)
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


#: Score gap below which two candidates of the same composition are a tie.
SIBLING_MARGIN = 0.12


def _flag_sibling_ties(result: PhaseSearchResult) -> None:
    """Say when the runner-up is the same compound at a different cell.

    The three turbostratic carbons differ only in interlayer spacing —
    3.36, 3.44 and 3.50 Å, putting the 002 at 26.5, 25.9 and 25.4°. A
    002 that is three degrees wide cannot choose between them, and
    reporting the winner alone would turn a coin toss into a measured
    spacing. Naming the tie is the honest result, and it is also the
    useful one: it says the number you want comes from refining the cell,
    not from the search.
    """
    for accepted in result.accepted:
        rivals = [
            m for m in result.rejected
            if m.crystal.formula == accepted.crystal.formula
            and m.crystal.name != accepted.crystal.name
            and accepted.score - m.score <= SIBLING_MARGIN
        ]
        if not rivals:
            continue
        names = ", ".join(f"{m.crystal.name} ({m.score:.2f})" for m in rivals[:3])
        result.warnings.append(
            f"{accepted.crystal.name} ({accepted.score:.2f}) empata con "
            f"{names}: es el mismo compuesto con otra celda y el "
            "difractograma no los separa. No informes el espaciado que sale "
            "de aquí — refina la celda, que para eso está el Rietveld"
        )


def _add_warnings(result: PhaseSearchResult, pattern: Pattern) -> None:
    """Say what a result implies about the measurement, not just the sample."""
    _flag_sibling_ties(result)
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
    "DETECTION_MARGIN",
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
    "strongest_below_threshold",
]
