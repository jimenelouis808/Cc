"""Why the calculated peaks do not reach the observed ones.

The complaint this module answers, verbatim: «los picos calculados no
alcanzan la altura del patrón experimental». It is the most common thing
to notice about a Rietveld plot and the one thing no R factor will tell
you, because on a pattern like the one it was measured on — a CVD carbon
on FeSe, peaks of 60 to 480 counts over a background of 545 — the peak
maxima are a few dozen points out of eight thousand. Rwp was 4.39 % with
GOF 1.045, the difference curve looked quiet, and the calculated peaks
came up at **68 %** of the observed heights.

Three different things produce that one appearance, they have opposite
meanings, and the figure cannot separate them:

**The comparison itself is biased.** The observed curve is noisy and the
calculated one is smooth. A smooth envelope passes through the MIDDLE of
the scatter, so the eye compares it against the highest point of the
peak, and the highest of *n* noisy points sits systematically above the
envelope. Measured on that pattern: comparing maxima gave a median ratio
of 0.68 and comparing envelopes gave 0.86 — more than half the apparent
deficit was this, and at the two weakest peaks (66 and 118 counts over
background) the whole of it was, because the expected noise maximum
alone is 44 counts.

**The profile can be too wide.** Then the AREA agrees and the height does
not, which is the arithmetic of area over width. Measured: at the
strongest peak the areas matched to 1.5 % while the height came to 90 %,
and the calculated curve sat ABOVE the observed on both flanks — the
intensity is there, it is in the wrong place.

**Or intensity is genuinely missing.** Then the area is short too, and
that is a statement about a phase's scale or about a phase that is not in
the model at all. Measured: at 43.7°, where the cementite group overlaps
the α-Fe 110, both the area and the height came to 76 %.

So this is not a check with a threshold, it is three questions asked in
order, and the answer is reported per peak. Nothing here changes a
refinement.

**On the envelope, and why it is not a smoothing.** The first version
estimated it with a short moving average and that was wrong in the
opposite direction: averaging over a third of the FWHM shaves the top of
the peak, so on a model that WAS the truth the ratio came back at 105 %.
A smoothing cannot be the reference for a height, because its own bias
is a function of the width being measured. What is used instead is
``observed maximum − expected maximum of the noise``, which needs no
smoothing and is unbiased by construction once the second term is
measured rather than assumed. It is also conservative: the expected
maximum is taken over the whole FWHM, while the maximum of the peak can
only fall in the few points near its top, so the correction is if
anything a little large — it errs towards calling a deficit noise.

**On the bias.** It is measured, not assumed: the expected maximum of the
noise comes from the residual itself, over peak-free windows of the same
length as the peak's. That needs no assumption about σ and none about
independence, which matters because a pattern stepped at 0.01° with an
instrumental width of 0.06° is oversampled four times over — its
consecutive points are correlated, so √N is right per point and wrong for
how far the maximum of a window travels. On the measured pattern √N gave
32 counts per point while the point-to-point scatter gave 16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..core.compat import trapezoid
from .pattern import Pattern
from .search import XRDPeak, find_peaks

#: Half-width of the window each peak is judged over, in FWHM. Wide
#: enough to contain the flanks where a too-wide profile shows itself,
#: narrow enough not to swallow the neighbour.
WINDOW_FWHM = 1.5

#: A peak whose observed height is below this many times the expected
#: noise maximum has no height to compare. It is not a weak result, it is
#: an absent question: the maximum of the noise over the same window is
#: the same size as the peak.
NOISE_MULTIPLE = 3.0

#: Area ratios inside ``1 ± AREA_TOLERANCE`` say the scale is right, so a
#: height that is short is the profile's shape.
AREA_TOLERANCE = 0.15

#: Height ratios inside ``1 ± HEIGHT_TOLERANCE`` agree. The envelope is
#: itself an estimate, so a tighter number would be reporting its own
#: noise.
HEIGHT_TOLERANCE = 0.10

#: Peak-free windows needed before the bias is taken from the residual
#: rather than from σ.
MIN_NOISE_WINDOWS = 8

#: A peak more than this much of whose window belongs to a stronger
#: neighbour is reported as the neighbour's, not judged on its own.
OVERLAP_LIMIT = 0.5


#: Blom's constant for the expected value of an order statistic. The
#: asymptotic √(2 ln n) form that this replaced is 0.12 to 0.21 low over
#: 5 ≤ n ≤ 1000 — checked against a seeded simulation — and low is the
#: wrong direction here: it would call noise a deficit.
BLOM = math.pi / 8.0


def expected_max_excess(points: int) -> float:
    """How far above zero the maximum of ``points`` standard normals sits.

    Blom's approximation to the expected value of the largest of *n*
    independent standard normals, good to about 0.03 from *n* = 2 up.

    It is only the fallback: when the pattern has a peak-free stretch the
    same quantity is measured from the residual instead, which is the
    better number because real diffractograms are oversampled —
    neighbouring points are correlated, and then no formula that assumes
    independence applies.
    """
    from scipy.special import ndtri

    n = max(int(points), 2)
    return float(ndtri((n - BLOM) / (n - 2.0 * BLOM + 1.0)))


@dataclass(frozen=True)
class PeakHeight:
    """One peak's observed and calculated height, and which of the three."""

    two_theta: float
    window: tuple[float, float]
    points: int
    background: float
    observed_max: float
    """Highest measured point above the background — the biased quantity."""
    observed_envelope: float
    """The same peak's height with its point-to-point scatter averaged
    out. This is what the calculated curve should be compared against."""
    calculated_max: float
    bias: float
    """Expected maximum of the noise over a window this long, above the
    envelope. The gap between :attr:`observed_max` and
    :attr:`observed_envelope` should be about this."""
    area_observed: float
    area_calculated: float
    verdict: str
    """``solapado``, ``ruido``, ``de acuerdo``, ``perfil``, ``sobra`` or
    ``escala``."""
    note: str

    @property
    def height_ratio(self) -> Optional[float]:
        """Calculated over observed ENVELOPE. ``None`` if there is no peak."""
        if self.observed_envelope <= 0.0:
            return None
        return float(self.calculated_max / self.observed_envelope)

    @property
    def raw_height_ratio(self) -> Optional[float]:
        """Calculated over the observed MAXIMUM — what the figure shows."""
        if self.observed_max <= 0.0:
            return None
        return float(self.calculated_max / self.observed_max)

    @property
    def area_ratio(self) -> Optional[float]:
        if self.area_observed <= 0.0:
            return None
        return float(self.area_calculated / self.area_observed)


@dataclass(frozen=True)
class HeightAgreement:
    """What the peak heights say, peak by peak and overall."""

    peaks: tuple[PeakHeight, ...]
    bias_origin: str
    """``residuo`` when the expected noise maximum was measured from the
    fit's own residual, ``sigma`` when there was no peak-free stretch long
    enough and it came from σ instead."""
    verdict: str
    """``de acuerdo``, ``perfil``, ``escala``, ``mezclado`` or ``ruido``."""
    notes: tuple[str, ...] = ()

    @property
    def comparable(self) -> tuple[PeakHeight, ...]:
        """The peaks that have a height at all."""
        return tuple(p for p in self.peaks
                     if p.verdict not in ("ruido", "solapado"))

    @property
    def median_height_ratio(self) -> Optional[float]:
        values = [p.height_ratio for p in self.comparable
                  if p.height_ratio is not None]
        return float(np.median(values)) if values else None

    @property
    def median_raw_height_ratio(self) -> Optional[float]:
        values = [p.raw_height_ratio for p in self.comparable
                  if p.raw_height_ratio is not None]
        return float(np.median(values)) if values else None

    @property
    def median_area_ratio(self) -> Optional[float]:
        values = [p.area_ratio for p in self.comparable
                  if p.area_ratio is not None]
        return float(np.median(values)) if values else None

    def describe(self) -> str:
        """The report, in the order the questions have to be asked."""
        lines = [
            "ALTURAS CALCULADAS FRENTE A OBSERVADAS",
            "──────────────────────────────────────",
        ]
        raw = self.median_raw_height_ratio
        envelope = self.median_height_ratio
        area = self.median_area_ratio
        if raw is not None and envelope is not None:
            lines.append(
                f"  contra el MÁXIMO medido:   {100 * raw:5.1f} %  "
                "(lo que se ve en la figura)")
            lines.append(
                f"  contra la ENVOLVENTE:      {100 * envelope:5.1f} %  "
                "(la comparación que no está sesgada)")
            if area is not None:
                lines.append(f"  en ÁREA:                   {100 * area:5.1f} %")
            lines.append("")
        lines.append(f"  {_VERDICTS[self.verdict]}")
        lines.append("")
        lines.append("   2θ        obs    envolv.    calc   alt.%   área%   qué dice")
        for peak in self.peaks:
            height = peak.height_ratio
            ratio = peak.area_ratio
            lines.append(
                f"  {peak.two_theta:7.3f} {peak.observed_max:8.0f} "
                f"{peak.observed_envelope:9.0f} {peak.calculated_max:8.0f} "
                f"{100 * height if height else float('nan'):6.0f}  "
                f"{100 * ratio if ratio else float('nan'):6.0f}   {peak.verdict}")
        lines.append("")
        lines.append(f"  el máximo esperado del ruido se ha medido del "
                     f"{self.bias_origin}")
        for note in self.notes:
            lines.append(f"  ⚠ {note}")
        return "\n".join(lines)


_VERDICTS = {
    "de acuerdo": (
        "las alturas concuerdan: la diferencia que se ve en la figura es el "
        "sesgo de comparar una curva suave con el máximo de unos puntos "
        "ruidosos"),
    "perfil": (
        "las ÁREAS concuerdan y las alturas no: el perfil calculado es más "
        "ancho que el medido. La intensidad está, está repartida de otra "
        "manera, y eso son las anchuras (U, V, W y la mezcla), no las "
        "fracciones en peso"),
    "escala": (
        "las áreas también se quedan cortas: falta intensidad, no anchura. "
        "Mira si a alguna fase le falta escala o si falta una fase entera"),
    "mezclado": (
        "unos picos son de perfil y otros de escala: míralos uno a uno, "
        "porque son dos arreglos distintos"),
    "sobra": (
        "el modelo pone más intensidad de la que hay: alguna fase va con "
        "demasiada escala, o hay una fase en el modelo que no es de esta "
        "muestra"),
    "ruido": (
        "ningún pico tiene altura que comparar: el máximo del propio ruido "
        "sobre una ventana del tamaño del pico es tan grande como el pico"),
}


def _overlapped(
    index: int,
    windows: Sequence[tuple[float, float]],
    heights: Sequence[float],
) -> Optional[int]:
    """Which stronger peak owns most of this one's window, if any.

    A detection on the flank of a much stronger neighbour has a window
    whose intensity is mostly the neighbour's, and then every ratio
    measured in it is a statement about the neighbour. It is reported and
    excluded rather than dropped, because «this row is its neighbour's»
    is itself the answer to «why does this peak look wrong».

    Measured: a synthetic case built to be purely a width problem
    reported its four peaks as «perfil» and a fifth, 0.34° away from one
    of them on its flank, as «sobra» with an area ratio of 2.25. Keeping
    it in the medians turned a clean verdict into «mezclado». The
    neighbour is not dropped on distance, which would be wrong on a real
    pattern where a 1.4°-wide cementite group legitimately sits 0.9° from
    the α-Fe 110.
    """
    low, high = windows[index]
    span = max(high - low, 1e-9)
    for other, (other_low, other_high) in enumerate(windows):
        if other == index or heights[other] <= heights[index]:
            continue
        shared = min(high, other_high) - max(low, other_low)
        if shared / span > OVERLAP_LIMIT:
            return other
    return None


def _noise_bias(
    two_theta: np.ndarray,
    residual: np.ndarray,
    sigma: np.ndarray,
    occupied: list[tuple[float, float]],
    points: int,
) -> tuple[float, str]:
    """Expected maximum of the noise over a window of ``points`` points.

    Measured from the residual over windows that hold no identified peak,
    which is the same thing the peaks' own windows contain plus the peak.
    No assumption about σ and none about independence: a diffractogram
    stepped well below its instrumental width has correlated neighbours,
    and then σ·√(2 ln n) is simply the wrong number.
    """
    points = max(int(points), 2)
    free = np.ones(two_theta.size, dtype=bool)
    for low, high in occupied:
        free &= ~((two_theta >= low) & (two_theta <= high))
    maxima: list[float] = []
    index = 0
    while index + points <= two_theta.size:
        if free[index:index + points].all():
            block = residual[index:index + points]
            maxima.append(float(block.max() - np.median(block)))
            index += points
        else:
            index += 1
    if len(maxima) >= MIN_NOISE_WINDOWS:
        return float(np.median(maxima)), "residuo"
    return float(np.median(sigma) * expected_max_excess(points)), "sigma"


def height_agreement(
    result,
    peaks: Optional[Sequence[XRDPeak]] = None,
    window_fwhm: float = WINDOW_FWHM,
) -> HeightAgreement:
    """Compare the calculated peak heights with the observed ones.

    Parameters
    ----------
    result:
        A :class:`~ramancarbon.xrd.rietveld.RietveldResult`.
    peaks:
        Which peaks to judge. Defaults to the pattern's own detected
        peaks, which is the right default because they are the ones
        somebody is looking at.
    window_fwhm:
        Half-width of each peak's window, in FWHM.

    Returns
    -------
    HeightAgreement
        Per peak and overall. Changes nothing about the refinement.
    """
    pattern: Pattern = result.pattern
    two_theta = pattern.two_theta
    observed = pattern.intensity
    calculated = np.asarray(result.calculated, dtype=float)
    background = np.asarray(result.background, dtype=float)
    found = list(peaks) if peaks is not None else list(find_peaks(pattern))
    sigma = np.asarray(
        pattern.sigma if pattern.sigma is not None
        else np.full(observed.shape, pattern.noise_estimate()), dtype=float)

    windows: list[tuple[float, float]] = []
    for peak in found:
        half = window_fwhm * max(float(peak.fwhm), pattern.step)
        windows.append((peak.two_theta - half, peak.two_theta + half))

    heights = [float(peak.height) for peak in found]
    checks: list[PeakHeight] = []
    origins: list[str] = []
    for index, (peak, (low, high)) in enumerate(zip(found, windows)):
        mask = (two_theta >= low) & (two_theta <= high)
        count = int(mask.sum())
        if count < 5:
            continue
        axis = two_theta[mask]
        above = observed[mask] - background[mask]
        model = calculated[mask] - background[mask]
        # The maximum can only land where the peak is near its top, so
        # the window the noise maximum is expected over is the FWHM and
        # not the whole judging window.
        top = max(3, int(round(float(peak.fwhm) / max(pattern.step, 1e-9))))
        bias, origin = _noise_bias(two_theta, observed - calculated, sigma,
                                   windows, top)
        origins.append(origin)
        envelope = max(float(above.max()) - bias, 0.0)
        owner = _overlapped(index, windows, heights)
        checks.append(_judge(
            peak=peak, window=(float(low), float(high)), points=count,
            owner=(float(found[owner].two_theta) if owner is not None else None),
            background=float(np.median(background[mask])),
            observed_max=float(above.max()), observed_envelope=envelope,
            calculated_max=float(model.max()), bias=bias,
            area_observed=float(trapezoid(above, axis)),
            area_calculated=float(trapezoid(model, axis)),
        ))

    notes: list[str] = []
    if not checks:
        return HeightAgreement(peaks=(), bias_origin="sigma", verdict="ruido",
                               notes=("no hay ningún pico detectado con "
                                      "ventana suficiente para juzgarlo",))
    # One origin for the whole report, and the weaker one wins: if any
    # window had to fall back to σ, saying «residuo» would overstate it.
    bias_origin = "sigma" if "sigma" in origins else "residuo"
    kinds = {check.verdict for check in checks
             if check.verdict not in ("ruido", "solapado")}
    if not kinds:
        verdict = "ruido"
    elif kinds == {"de acuerdo"}:
        verdict = "de acuerdo"
    elif kinds <= {"de acuerdo", "perfil"}:
        verdict = "perfil"
    elif kinds <= {"de acuerdo", "escala"}:
        verdict = "escala"
    elif kinds <= {"de acuerdo", "sobra"}:
        verdict = "sobra"
    else:
        verdict = "mezclado"

    shared = [c for c in checks if c.verdict == "solapado"]
    if shared:
        notes.append(
            f"{len(shared)} pico(s) caen dentro de la ventana de un vecino "
            "más intenso y no se juzgan por su cuenta: "
            + ", ".join(f"{c.two_theta:.2f}°" for c in shared))
    quiet = [c for c in checks if c.verdict == "ruido"]
    if quiet:
        notes.append(
            f"{len(quiet)} de {len(checks)} pico(s) no llegan a "
            f"{NOISE_MULTIPLE:g} veces el máximo esperado del ruido, así que "
            "de su altura no se puede decir nada: "
            + ", ".join(f"{c.two_theta:.2f}°" for c in quiet))
    if bias_origin == "sigma":
        notes.append(
            "no quedaba patrón libre de picos para medir el máximo del ruido, "
            "así que viene de σ y supone puntos independientes. Un patrón "
            "muy sobremuestreado los tiene correlacionados, y entonces esta "
            "estimación es alta: se equivoca llamando ruido a un déficit")
    if result.background_share > 0.5:
        notes.append(
            f"el fondo es el {100 * result.background_share:.0f} % del patrón "
            "calculado, así que los picos son una parte pequeñísima de lo que "
            "minimiza el Rwp: ningún factor R iba a quejarse de estas alturas")
    return HeightAgreement(peaks=tuple(checks), bias_origin=bias_origin,
                           verdict=verdict, notes=tuple(notes))


def _judge(
    peak: XRDPeak,
    window: tuple[float, float],
    points: int,
    background: float,
    observed_max: float,
    observed_envelope: float,
    calculated_max: float,
    bias: float,
    area_observed: float,
    area_calculated: float,
    owner: Optional[float] = None,
) -> PeakHeight:
    """The three questions, in order, for one peak."""
    if owner is not None:
        verdict = "solapado"
        note = (f"más de la mitad de su ventana es del pico de {owner:.2f}°, "
                "que es más intenso: lo que se mida aquí habla de ése")
    elif observed_envelope < NOISE_MULTIPLE * max(bias, 1e-9):
        verdict = "ruido"
        note = (f"la envolvente observada ({observed_envelope:.0f}) no llega a "
                f"{NOISE_MULTIPLE:g}× el máximo esperado del ruido "
                f"({bias:.0f}) sobre esta ventana: no hay altura que comparar")
    else:
        height = calculated_max / observed_envelope
        ratio = (area_calculated / area_observed) if area_observed > 0 else None
        if abs(height - 1.0) <= HEIGHT_TOLERANCE:
            verdict = "de acuerdo"
            note = (f"la altura concuerda al {100 * height:.0f} %; contra el "
                    f"máximo medido sale {100 * calculated_max / observed_max:.0f} "
                    "% porque el máximo de unos puntos ruidosos está por "
                    f"encima de su envolvente, aquí {bias:.0f} cuentas")
        elif ratio is not None and abs(ratio - 1.0) <= AREA_TOLERANCE:
            verdict = "perfil"
            note = (f"el área concuerda al {100 * ratio:.0f} % y la altura "
                    f"solo al {100 * height:.0f} %: el pico calculado es más "
                    "ancho que el medido")
        elif ratio is not None and ratio > 1.0 + AREA_TOLERANCE:
            verdict = "sobra"
            note = (f"el área calculada es el {100 * ratio:.0f} % de la "
                    "medida: aquí el modelo pone MÁS intensidad de la que "
                    "hay, que es una fase con demasiada escala o una fase "
                    "que no es de esta muestra")
        else:
            verdict = "escala"
            note = (f"el área sale al {100 * ratio:.0f} % y la altura al "
                    f"{100 * height:.0f} %: falta intensidad, no anchura"
                    if ratio is not None else
                    f"la altura sale al {100 * height:.0f} % y el área "
                    "observada no es positiva")
    return PeakHeight(
        two_theta=float(peak.two_theta), window=window, points=points,
        background=background, observed_max=observed_max,
        observed_envelope=observed_envelope, calculated_max=calculated_max,
        bias=bias, area_observed=area_observed,
        area_calculated=area_calculated, verdict=verdict, note=note,
    )


__all__ = [
    "AREA_TOLERANCE",
    "HEIGHT_TOLERANCE",
    "HeightAgreement",
    "MIN_NOISE_WINDOWS",
    "NOISE_MULTIPLE",
    "PeakHeight",
    "WINDOW_FWHM",
    "expected_max_excess",
    "height_agreement",
]
