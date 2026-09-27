"""How small a band could have been and still been found.

A phase report says which of a phase's lines were seen. It does not say
what the *absence* of the others is worth, and that is usually the
question. On a 532 nm spectrum of carbon grown over FeSe the report
identified cementite from its 215 and 282 cm⁻¹ lines and said nothing
about FeSe — while the Rietveld refinement of the same sample put FeSe-T
at 2.7 % by weight. Is that a disagreement between the two techniques, or
did Raman simply never have the sensitivity? Without a number the report
can only shrug, which is what its note "seeing it is informative, NOT
seeing it proves nothing" amounts to.

This module turns the shrug into a limit: the smallest peak height that,
placed at the band's catalogued position, this program's own peak finder
would have reported. That last clause is the whole design. The limit is
computed by *injecting* a band of height h into the spectrum and asking
:func:`~ramancarbon.core.peaks.find_peaks` whether it finds it, then
bisecting on h — not by inverting a formula for the significance. An
inverted formula is a second implementation of the detector that drifts
away from the first one the moment either changes, and a detection limit
that does not match the detector is worse than none: it invites exactly
the conclusion it cannot support.

The same argument was made for :mod:`ramancarbon.xrd.detection_limit`,
which runs the diffraction significance test backwards through
``calculate_pattern`` for the same reason.

What the number depends on, and therefore what has to be quoted with it:

- **The assumed width.** A limit is a limit on a band of some shape. A
  narrow line is easier to see against noise than a broad one of the same
  height, so ``fwhm`` is an argument with no safe default; 15 cm⁻¹ is a
  common Raman linewidth for a crystalline inorganic phase and is what
  this module uses when nothing is said, and it says so in the summary.
- **The detection threshold.** The same one the peak finder uses, so the
  answer is about this program and not about an idealised experiment.
- **The local noise.** Measured from the spectrum, where the band would
  be — not a global number. The noise at 200 cm⁻¹ and at 1400 cm⁻¹ of the
  same trace differ by a third.

A limit on a *height* is not a limit on a weight fraction, and this
module does not pretend otherwise. Converting one to the other needs the
Raman cross-sections of both phases at the excitation used, and for the
iron carbides and selenides those are not known well enough to divide by.
What the height limit does support is a comparison *within one spectrum*:
"a FeSe band would have been found at 6 % of the height of the cementite
band", which is a statement about this measurement and needs no
cross-section at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..core.peaks import PeakMeasurement, find_peaks
from ..core.spectrum import Spectrum
from .phases import Phase, load_phases

#: Linewidth assumed for the injected band, in cm⁻¹, when none is given.
DEFAULT_FWHM = 15.0

#: Largest height the search considers, as a multiple of the spectrum's
#: own range. Beyond this the band would be the strongest thing in the
#: spectrum and "it would have been seen" stops being informative.
MAX_HEIGHT_FACTOR = 2.0

#: Bisection steps. Twenty halvings take the bracket to a millionth of
#: its width, which is far finer than the number deserves.
BISECTION_STEPS = 20


@dataclass
class BandLimit:
    """What one catalogued line of a phase is worth as evidence."""

    position: float
    assignment: str
    relative: float
    """Catalogue intensity, relative to the phase's strongest line."""
    detected: bool = False
    measured_height: Optional[float] = None
    limit_height: Optional[float] = None
    """Smallest height the peak finder would have reported, or ``None``
    when the question could not be asked."""
    reason: str = ""
    """Why there is no limit, when there is none."""
    obscured_by: Optional[float] = None
    """Position of a band already in the spectrum that sits in the way.

    A weak limit has two very different causes and the user has to be
    able to tell them apart. Either the noise is high — nothing to be
    done but count longer — or another band is sitting on top of the
    window, and then no amount of counting helps and the line is simply
    not usable on this sample. On the FeSe spectrum hematite's strongest
    line at 292 cm⁻¹ could only be bounded at 137 counts, not because the
    noise is anywhere near that but because cementite's 282 is right
    there. Reported as "no limit worth having" rather than as a limit."""

    def __str__(self) -> str:
        if self.detected:
            return (f"{self.position:7.1f} cm⁻¹  DETECTADA"
                    + (f" (altura {self.measured_height:.1f})"
                       if self.measured_height is not None else ""))
        if self.limit_height is None:
            return f"{self.position:7.1f} cm⁻¹  sin límite: {self.reason}"
        text = (f"{self.position:7.1f} cm⁻¹  no detectada; se habría visto "
                f"por encima de {self.limit_height:.1f}")
        if self.obscured_by is not None:
            text += (f" — pero la banda de {self.obscured_by:.0f} cm⁻¹ está "
                     "encima: el límite lo pone ella, no el ruido")
        return text


@dataclass
class PhaseLimit:
    """The limits for every line of one phase, and what they add up to."""

    phase: Phase
    bands: list[BandLimit] = field(default_factory=list)
    reference_position: Optional[float] = None
    reference_height: Optional[float] = None
    """A band actually present in the spectrum, used as the scale."""
    fwhm: float = DEFAULT_FWHM
    silent: bool = False
    """The phase has no first-order Raman modes at all."""

    @property
    def detected(self) -> bool:
        return any(band.detected for band in self.bands)

    @property
    def best_limit(self) -> Optional[BandLimit]:
        """The line that constrains the phase hardest.

        The most informative absence, relative to how strong the line
        should have been: a missing line the catalogue calls weak says
        much less than a missing strong one, so the limit is divided by
        the catalogued relative intensity before they are compared.
        """
        candidates = [b for b in self.bands
                      if not b.detected and b.limit_height is not None
                      and b.relative > 0 and b.obscured_by is None]
        if not candidates:
            return None
        return min(candidates, key=lambda b: b.limit_height / b.relative)

    @property
    def anchor(self) -> Optional[BandLimit]:
        """The detected line the rest of the phase is predicted from.

        The strongest one, because the catalogue's relative intensities
        are quoted against the strongest line and dividing by a weak one
        multiplies its error into every prediction.
        """
        seen = [b for b in self.bands if b.detected
                and b.measured_height is not None and b.relative > 0]
        return max(seen, key=lambda b: b.relative) if seen else None

    @property
    def excluded_by(self) -> list[tuple[BandLimit, float]]:
        """Lines that rule the phase out, with the height they should have.

        This is where a limit stops being a number and becomes an
        argument. One line of a phase landing on a peak proves nothing —
        three different phases in this catalogue have a line within
        8 cm⁻¹ of 393 cm⁻¹. But if that line really belonged to the
        phase, the catalogue says how tall its *other* lines would be,
        and those are lines whose absence has just been quantified.

        The prediction rests on the catalogue's relative intensities
        holding at this excitation, which they only roughly do, so the
        gap has to be large before it counts: see
        :data:`EXCLUSION_MARGIN`, which exists because at a margin of one
        this check confidently ruled out the phase the diffraction
        refinement puts at 16 % by weight.

        Lines whose limit was set by a neighbouring band rather than by
        the noise are left out, because there the prediction and the
        limit are not comparable.
        """
        anchor = self.anchor
        if anchor is None or not anchor.measured_height:
            return []
        out: list[tuple[BandLimit, float]] = []
        for band in self.bands:
            if band.detected or band.limit_height is None:
                continue
            if band.obscured_by is not None or band.relative <= 0:
                continue
            expected = (band.relative / anchor.relative) * anchor.measured_height
            if expected > EXCLUSION_MARGIN * band.limit_height:
                out.append((band, expected))
        return sorted(out, key=lambda item: -(item[1] / item[0].limit_height))

    def verdict(self) -> str:
        """One sentence: what the limits let you conclude."""
        if self.silent:
            return "Raman no puede ver esta fase en absoluto."
        excluded = self.excluded_by
        if excluded:
            band, expected = excluded[0]
            anchor = self.anchor
            return (
                f"muy improbable: si el pico de {anchor.position:.0f} cm⁻¹ "
                f"fuera suyo, su línea de {band.position:.0f} cm⁻¹ debería "
                f"medir {expected:.0f} y se habría detectado por encima de "
                f"{band.limit_height:.0f} — un factor "
                f"{expected / band.limit_height:.0f}. Supone que las "
                "intensidades relativas del catálogo valgan a este láser"
            )
        anchor = self.anchor
        if self.detected and anchor is not None:
            confirmed = sum(1 for b in self.bands if b.detected)
            if confirmed >= 2:
                return f"compatible: {confirmed} líneas presentes"
            # One line is one line, whatever else the catalogue lists. Say
            # how close the others came to deciding it, because "cannot be
            # bounded" and "bounded, and it was not enough" are different
            # states and the second one is progress.
            tightest = 0.0
            for band in self.bands:
                if band.detected or band.limit_height is None:
                    continue
                if band.obscured_by is not None or band.relative <= 0:
                    continue
                expected = ((band.relative / anchor.relative)
                            * (anchor.measured_height or 0.0))
                tightest = max(tightest, expected / max(band.limit_height, 1e-9))
            if tightest <= 0.0:
                return ("una sola línea presente y las demás no se pueden "
                        "acotar en este espectro: ni confirmada ni descartada")
            return (
                f"una sola línea presente; sus demás líneas se han acotado, "
                f"pero la más exigente sólo llega a un factor "
                f"{tightest:.1f} sobre su límite, por debajo del "
                f"{EXCLUSION_MARGIN:.0f} que haría falta para descartarla: "
                "ni confirmada ni descartada"
            )
        best = self.best_limit
        if best is None:
            return ("no se puede decir nada: ninguna de sus líneas es "
                    "medible en este espectro")
        return (f"no detectada; lo más que se puede afirmar es que su línea "
                f"de {best.position:.0f} cm⁻¹ está por debajo de "
                f"{best.limit_height:.0f}")

    def summary(self) -> str:
        lines = [f"{self.phase.label} [{self.phase.key}]"]
        if self.silent:
            lines.append("  no tiene modos Raman de primer orden: esta "
                         "técnica no puede verla ni confirmarla. Su ausencia "
                         "en el espectro no es evidencia de nada")
            lines.append("  → " + self.verdict())
            return "\n".join(lines)
        lines.append(f"  banda inyectada de FWHM {self.fwhm:.0f} cm⁻¹, "
                     "umbral el del buscador de picos del programa")
        for band in self.bands:
            lines.append("  " + str(band))
        best = self.best_limit
        if best is not None and self.reference_height:
            fraction = 100.0 * best.limit_height / self.reference_height
            lines.append(
                f"  la línea que más restringe es la de {best.position:.0f} "
                f"cm⁻¹: se habría detectado con el {fraction:.1f} % de la "
                f"altura de la banda de {self.reference_position:.0f} cm⁻¹")
        lines.append("  → " + self.verdict())
        return "\n".join(lines)


def _inject(spectrum: Spectrum, centre: float, fwhm: float,
            height: float) -> Spectrum:
    """The spectrum with one Lorentzian added at ``centre``.

    Lorentzian and not Gaussian because it is the conservative choice
    here: its wings put a larger share of the area outside the window, so
    the peak finder has less to work with and the limit comes out
    slightly higher. A limit that errs upwards is the one you want — it
    makes the absence of a band say less, not more.
    """
    x = np.asarray(spectrum.shift, dtype=float)
    half = max(fwhm, 1e-6) / 2.0
    band = height / (1.0 + ((x - centre) / half) ** 2)
    return Spectrum(
        shift=x, intensity=np.asarray(spectrum.intensity, dtype=float) + band,
        laser_nm=spectrum.laser_nm, name=spectrum.name,
        metadata=dict(spectrum.metadata),
    )


def _found_at(spectrum: Spectrum, centre: float, fwhm: float,
              tolerance: float, **options) -> bool:
    """Whether the peak finder reports a peak at ``centre``."""
    pad = max(4.0 * fwhm, 3.0 * tolerance)
    window = (centre - pad, centre + pad)
    if window[0] < spectrum.range[0] or window[1] > spectrum.range[1]:
        window = (max(window[0], spectrum.range[0]),
                  min(window[1], spectrum.range[1]))
    try:
        found = find_peaks(spectrum, window=window, **options)
    except ValueError:
        return False
    return any(abs(peak.position - centre) <= tolerance for peak in found)


def noise_floor_limit(
    spectrum: Spectrum,
    centre: float,
    fwhm: float = DEFAULT_FWHM,
    tolerance: float = 8.0,
    seed: int = 20260927,
    **options,
) -> Optional[float]:
    """The limit this spectrum's noise alone would impose at ``centre``.

    A limit is a number; what a reader wants to know is whether it is a
    number about *sensitivity* or about a band sitting in the way, and
    those call for opposite actions — count longer, or give up on that
    line for this sample. Comparing the real limit against the limit on a
    trace of the same noise and nothing else separates them, and it does
    so without a threshold on "how close is too close": two lines thirty
    wavenumbers apart may or may not interfere depending on how wide and
    how tall they are, and this measures the interference instead of
    guessing it from the gap.

    The surrogate keeps the real axis, so the point spacing and the
    window length are the ones the detector will actually see, and it is
    seeded, so the same spectrum gives the same answer twice.
    """
    x = np.asarray(spectrum.shift, dtype=float)
    noise = spectrum.local_noise()
    near = np.abs(x - centre) <= max(4.0 * fwhm, 3.0 * tolerance)
    sigma = float(np.median(noise[near])) if near.any() else float(np.median(noise))
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    rng = np.random.default_rng(seed)
    surrogate = Spectrum(
        shift=x, intensity=rng.normal(0.0, sigma, x.shape),
        laser_nm=spectrum.laser_nm, name="ruido",
    )
    return band_limit(surrogate, centre, fwhm=fwhm, tolerance=tolerance,
                      **options)


def band_limit(
    spectrum: Spectrum,
    centre: float,
    fwhm: float = DEFAULT_FWHM,
    tolerance: float = 8.0,
    **options,
) -> Optional[float]:
    """Smallest height at ``centre`` the peak finder would have reported.

    Returns ``None`` when even a band as tall as the spectrum's own range
    would not be reported — which happens where the window falls off the
    end of the measurement, and is a refusal rather than an answer.

    Parameters
    ----------
    spectrum:
        Baseline-corrected spectrum. Not corrected here, because the
        caller's baseline is part of what is being characterised: a limit
        computed on a differently flattened trace is a limit on a
        different measurement.
    centre:
        Where the band would be, in cm⁻¹.
    fwhm:
        Assumed linewidth. See the module docstring: the limit is a limit
        on a band of this width and the number is meaningless without it.
    tolerance:
        How far the recovered peak may sit from ``centre`` and still
        count as the same band.
    **options:
        Passed to :func:`~ramancarbon.core.peaks.find_peaks`, so a caller
        who tightened the threshold for the analysis gets a limit under
        that same threshold.
    """
    span = float(np.ptp(np.asarray(spectrum.intensity, dtype=float)))
    ceiling = MAX_HEIGHT_FACTOR * max(span, 1e-9)
    if not _found_at(_inject(spectrum, centre, fwhm, ceiling), centre, fwhm,
                     tolerance, **options):
        return None
    if _found_at(spectrum, centre, fwhm, tolerance, **options):
        return 0.0
    low, high = 0.0, ceiling
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if _found_at(_inject(spectrum, centre, fwhm, middle), centre, fwhm,
                     tolerance, **options):
            high = middle
        else:
            low = middle
    return high


def detection_limit(
    spectrum: Spectrum,
    phase: str | Phase,
    peaks: Optional[Sequence[PeakMeasurement]] = None,
    fwhm: float = DEFAULT_FWHM,
    tolerance: float = 8.0,
    reference: Optional[float] = None,
    directory: Optional[str] = None,
    **options,
) -> PhaseLimit:
    """What the absence of a phase's lines is worth, line by line.

    Parameters
    ----------
    spectrum:
        Baseline-corrected spectrum.
    phase:
        A :class:`~ramancarbon.analysis.phases.Phase` or its key.
    peaks:
        Already-detected peaks, to decide which lines were seen. Found
        with the same options when not given.
    fwhm:
        Assumed linewidth of the band that is not there.
    tolerance:
        Match window, in cm⁻¹.
    reference:
        Position of a band to use as the scale, normally one belonging to
        a phase that *was* identified. Defaults to the tallest detected
        peak, so the limit is quoted against the strongest thing in the
        spectrum.
    directory:
        Phase database directory, for a phase given by key.

    Raises
    ------
    ValueError
        If the phase key is not in the database.
    """
    if isinstance(phase, str):
        catalogue = {p.key: p for p in load_phases(directory)}
        if phase not in catalogue:
            raise ValueError(
                f"«{phase}» no está en la base de fases; claves disponibles: "
                + ", ".join(sorted(catalogue)[:12]) + "…")
        phase = catalogue[phase]

    if phase.raman_silent:
        return PhaseLimit(phase=phase, bands=[], fwhm=fwhm, silent=True)

    if peaks is None:
        peaks = find_peaks(spectrum, **options)
    peaks = list(peaks)

    if reference is None and peaks:
        tallest = max(peaks, key=lambda p: p.height)
        reference, reference_height = tallest.position, tallest.height
    elif reference is not None:
        near = [p for p in peaks if abs(p.position - reference) <= tolerance]
        reference_height = near[0].height if near else None
    else:
        reference_height = None

    low, high = spectrum.range
    limits: list[BandLimit] = []
    for band in phase.bands:
        entry = BandLimit(position=band.position, assignment=band.assignment,
                          relative=band.relative)
        hit = [p for p in peaks if abs(p.position - band.position) <= tolerance]
        if hit:
            entry.detected = True
            entry.measured_height = max(p.height for p in hit)
            limits.append(entry)
            continue
        if not low <= band.position <= high:
            # Saying "not detected" about a line outside the measured
            # range would be the single most misleading thing this module
            # could do: it is not absent, it was never looked for.
            entry.reason = (f"fuera del intervalo medido "
                            f"({low:.0f}–{high:.0f} cm⁻¹)")
            limits.append(entry)
            continue
        value = band_limit(spectrum, band.position, fwhm=fwhm,
                           tolerance=tolerance, **options)
        if value is None:
            entry.reason = ("ni una banda tan alta como todo el espectro se "
                            "detectaría aquí: demasiado cerca del borde o del "
                            "ruido")
        entry.limit_height = value
        # Proximity alone does not obscure anything: a tiny feature beside
        # the window costs the search nothing. What obscures is intensity
        # at least as tall as the band being looked for, so the test is
        # made against the limit that was just computed.
        #
        # Against the SPECTRUM and not against the peak list, because the
        # two do not agree on height. A peak's reported height depends on
        # how it was measured — above a local baseline, above zero, at the
        # parabolic vertex — and on the FeSe spectrum that disagreement
        # was enough to leave a limit of 171 counts unflagged with a band
        # of very nearly that height sitting on it, which reads as a
        # sensitivity and is nothing of the sort.
        if value is not None:
            floor = noise_floor_limit(spectrum, band.position, fwhm=fwhm,
                                      tolerance=tolerance, **options)
            if floor and value > OBSCURED_RATIO * floor:
                pad = max(4.0 * fwhm, 3.0 * tolerance)
                x = np.asarray(spectrum.shift, dtype=float)
                y = np.asarray(spectrum.intensity, dtype=float)
                near = ((np.abs(x - band.position) <= pad)
                        & (np.abs(x - band.position) > 0.5 * fwhm))
                if near.any():
                    entry.obscured_by = float(
                        x[near][int(np.argmax(y[near]))])
        limits.append(entry)

    return PhaseLimit(phase=phase, bands=limits, fwhm=fwhm,
                      reference_position=reference,
                      reference_height=reference_height)


#: How much worse than the noise alone a limit has to be before it is
#: blamed on a neighbouring band rather than on sensitivity.
#:
#: Two. The separation is not delicate: on the FeSe spectrum the lines
#: sitting under the 215 and 282 cm⁻¹ bands come out at 120 to 190 counts
#: against a noise floor near 25, while every line in clear air lands
#: between 10 and 25 — a gap of an order of magnitude with nothing in it.
OBSCURED_RATIO = 2.0

#: How far a predicted height must exceed its own limit before the
#: absence of that line is allowed to count against the phase.
#:
#: Five, and the number has a scar on it. At a margin of one — predicted
#: above the limit, therefore excluded — the first thing this check did on
#: a real spectrum was rule out cementite, the one phase the Rietveld
#: refinement of the same sample puts at 16.4 % by weight and the one
#: whose two strongest lines are plainly present at 215 and 282 cm⁻¹. It
#: did so from the missing 685 cm⁻¹ line, predicted at 59 counts against a
#: limit of 19: a factor of three.
#:
#: The fault is in the premise, not the arithmetic. Predicting one band's
#: height from another's needs the catalogue's relative intensities to
#: hold at *this* excitation and *this* geometry, and Raman relative
#: intensities do not travel that way — they move with the laser through
#: resonance, with crystallite orientation, and with whatever the band is
#: sitting on. A factor of three between catalogue and measurement is
#: ordinary. A factor of eighteen, which is what actually excludes anatase
#: on the same spectrum, is not.
#:
#: So the margin buys back the premise's error, and even then the wording
#: stays short of a verdict.
EXCLUSION_MARGIN = 5.0

#: Most phases :func:`absence_limits` will bound in one call.
#:
#: Not a performance guard for its own sake. A report that bounds thirty
#: phases is a report nobody reads, and the ones worth bounding are the
#: few the scan already raised as possibilities.
MAX_PHASES = 8


def absence_limits(
    spectrum: Spectrum,
    report,
    extra: Sequence[str] = (),
    peaks: Optional[Sequence[PeakMeasurement]] = None,
    fwhm: float = DEFAULT_FWHM,
    tolerance: float = 8.0,
    directory: Optional[str] = None,
    **options,
) -> list[PhaseLimit]:
    """Bound the phases a scan raised and could not settle.

    The phase scan ends in three kinds of open question: a peak it could
    not explain with a list of near misses, a phase matched on one line
    and not corroborated, and a family where one polymorph was named and
    the others were not. All three are answered by the same number — how
    much of the phase could be there without having been seen — and none
    of them was answered at all.

    Parameters
    ----------
    spectrum:
        Baseline-corrected spectrum, the one the peaks came from.
    report:
        A :class:`~ramancarbon.analysis.phases.PhaseReport`.
    extra:
        Phase keys to bound whatever the scan said, for the question the
        user actually has. A Rietveld refinement that puts FeSe-T at
        2.7 % raises "should Raman have seen that?", and the Raman scan
        has no way to know it was asked.
    peaks:
        Already-detected peaks. Found with the same options when absent.

    Returns
    -------
    list[PhaseLimit]
        In the order they were chosen, identified phases last.
    """
    catalogue = {phase.key: phase for phase in load_phases(directory)}
    by_label = {phase.label: phase.key for phase in catalogue.values()}

    wanted: list[str] = []

    def want(key: Optional[str]) -> None:
        if key and key in catalogue and key not in wanted:
            wanted.append(key)

    for key in extra:
        want(key)
    for entry in getattr(report, "identifications", []):
        if not getattr(entry, "corroborated", False):
            want(getattr(entry.phase, "key", None))
    for candidates in getattr(report, "near_misses", {}).values():
        for label, _position, _distance in candidates:
            want(by_label.get(label))

    identified = {getattr(entry.phase, "key", None)
                  for entry in getattr(report, "identifications", [])
                  if getattr(entry, "corroborated", False)}
    for key in identified:
        want(key)

    if peaks is None:
        peaks = find_peaks(spectrum, **options)
    peaks = list(peaks)
    reference = (max(peaks, key=lambda q: q.height).position
                 if peaks else None)

    return [
        detection_limit(spectrum, catalogue[key], peaks=peaks, fwhm=fwhm,
                        tolerance=tolerance, reference=reference,
                        directory=directory, **options)
        for key in wanted[:MAX_PHASES]
    ]


__all__ = [
    "BISECTION_STEPS",
    "DEFAULT_FWHM",
    "MAX_HEIGHT_FACTOR",
    "MAX_PHASES",
    "OBSCURED_RATIO",
    "BandLimit",
    "PhaseLimit",
    "absence_limits",
    "band_limit",
    "detection_limit",
    "noise_floor_limit",
]
