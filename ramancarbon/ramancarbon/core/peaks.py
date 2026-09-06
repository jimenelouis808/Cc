"""Finding peaks and measuring them without fitting.

Two different jobs live here and they should not be confused:

* :func:`find_peaks` locates candidate bands. Its output seeds the
  deconvolution and drives the band assignment; it is not a measurement.
* :func:`measure_peak` extracts a position, height, FWHM and area from the
  data around a maximum by interpolation. It is fast, model-free and good
  enough for a survey, but for overlapping bands — which in the 1300–1650
  cm⁻¹ region of any real carbon sample means *all* of them — only a
  proper deconvolution separates D from D3 or G from D'. Anything this
  module returns for that window is marked ``model_free=True`` so the
  report can say where the number came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.signal import find_peaks as _scipy_find_peaks
from scipy.signal import peak_prominences
from scipy.signal import peak_widths as _peak_widths

from .spectrum import Spectrum


@dataclass
class PeakMeasurement:
    """A band measured directly off the data, without a lineshape model."""

    position: float
    """Raman shift of the maximum, cm⁻¹, refined by parabolic interpolation."""

    height: float
    """Intensity above the local background at ``position``."""

    fwhm: Optional[float]
    """Full width at half maximum in cm⁻¹, or ``None`` if the half-maximum
    crossings fall outside the search window (a shoulder rather than a peak)."""

    area: float
    """Trapezoidal area over ±``window`` around the peak, background removed."""

    prominence: float
    """How far the peak rises above the higher of its two flanking valleys.
    The most useful single number for telling a real band from a ripple."""

    snr: float
    """Height divided by the spectrum's noise σ. Informative, but on its own a
    poor detection criterion: over a few hundred points, 3–4σ maxima occur by
    chance. Use :attr:`significance` to decide whether a peak is real."""

    significance: float = 0.0
    """Matched-filter significance, ``h·sqrt(FWHM/step)/σ``.

    A peak's detectability depends on how many points it spans, not only on
    how tall it is: a broad band of modest height is far more significant
    than a one-pixel spike of the same height. This is the quantity
    :func:`find_peaks` thresholds on."""

    model_free: bool = True
    """Always True here; kept so downstream reports can mix these with
    fitted components and still say which is which."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        w = f"{self.fwhm:.1f}" if self.fwhm else "n/a"
        return f"PeakMeasurement({self.position:.1f} cm⁻¹, h={self.height:.3g}, FWHM={w})"


def find_peaks(
    spectrum: Spectrum,
    min_significance: float = 18.0,
    min_prominence_fraction: float = 0.02,
    min_distance_cm: float = 8.0,
    min_fwhm_cm: float = 4.0,
    window: Optional[tuple[float, float]] = None,
) -> list[PeakMeasurement]:
    """Locate candidate bands.

    Position comes from a parabolic vertex through the maximum and its two
    neighbours; height and width are referred to the peak's own prominence
    base, i.e. the higher of the two valleys flanking it. That choice
    matters for shoulders: measuring D’ against a straight line drawn across
    ±40 cm⁻¹ would subtract most of the G band's flank along with the
    background and report a width several times too small.

    Parameters
    ----------
    spectrum:
        Baseline-corrected spectrum. Running this on an uncorrected
        spectrum works, but a rising fluorescence background suppresses
        prominence on the low-shift side and hides the RBM region.
    min_significance:
        Detection threshold on the matched-filter significance
        ``h·sqrt(FWHM/step)/σ``, automatically scaled up for longer search
        windows.

        A plain "height above 3σ" criterion does not work here, and using
        one is why an earlier version of this function reported four
        non-existent RBM peaks in every graphene spectrum. Searching several
        hundred points for a maximum is several hundred independent trials,
        and the largest of 320 Gaussian samples is about 3σ *by
        construction* — so a 3σ cut finds a "peak" in pure noise almost
        every time.

        The default of 18 was calibrated on 400 synthetic pure-noise windows
        of 320 points: their 99th-percentile significance is 18.4, while a
        real band only 3σ tall with a normal 10 cm⁻¹ width scores above 21.
        The threshold is scaled by ``sqrt(ln n / ln 320)`` to keep the
        false-alarm rate near 1 % per search regardless of how many points
        are being searched — the standard look-elsewhere correction, since
        the expected maximum of *n* samples grows as ``sqrt(2 ln n)``.
    min_prominence_fraction:
        Reject peaks whose prominence is below this fraction of the largest
        prominence in the spectrum. Filters ripples on the flanks of the G
        band without needing an absolute threshold.
    min_distance_cm:
        Minimum separation between accepted maxima, in cm⁻¹. Below ~8 cm⁻¹
        two maxima are not resolvable in a typical carbon spectrum and one
        of them is noise.
    min_fwhm_cm:
        Reject anything narrower than this. No carbon Raman band is
        narrower than a few cm⁻¹ — the sharpest RBMs are ~5 cm⁻¹ — so a
        one-or-two-pixel spike that survived despiking is rejected here.
    window:
        Optional ``(low, high)`` to restrict the search.

    Returns
    -------
    list[PeakMeasurement]
        Sorted by ascending Raman shift.
    """
    work = spectrum.crop(*window) if window else spectrum
    y = work.intensity
    x = work.shift
    if y.size < 5:
        return []

    sigma = work.noise_estimate()
    step = work.step
    distance = max(1, int(round(min_distance_cm / max(step, 1e-9))))
    threshold = min_significance * _trials_factor(y.size)

    indices, _ = _scipy_find_peaks(y, distance=distance)
    if indices.size == 0:
        return []

    prominences, left_bases, right_bases = peak_prominences(y, indices)
    if prominences.size and prominences.max() > 0:
        keep = prominences >= min_prominence_fraction * prominences.max()
    else:
        keep = np.ones(indices.size, dtype=bool)
    indices = indices[keep]
    prominences = prominences[keep]
    left_bases = left_bases[keep]
    right_bases = right_bases[keep]
    if indices.size == 0:
        return []

    # Width at half prominence. rel_height=0.5 measures down from the peak
    # towards its own base rather than towards zero, which is what "FWHM"
    # means for a band sitting on the flank of a larger one.
    widths, _, left_ips, right_ips = _peak_widths(y, indices, rel_height=0.5)

    peaks: list[PeakMeasurement] = []
    for k, idx in enumerate(indices):
        position, apex = _parabolic_vertex(x, y, int(idx))
        base = max(float(y[left_bases[k]]), float(y[right_bases[k]]))
        height = apex - base
        if height <= 0:
            continue
        left_cm = float(np.interp(left_ips[k], np.arange(x.size), x))
        right_cm = float(np.interp(right_ips[k], np.arange(x.size), x))
        fwhm = right_cm - left_cm
        if not np.isfinite(fwhm) or fwhm < min_fwhm_cm:
            continue
        significance = (
            height * np.sqrt(fwhm / max(step, 1e-9)) / sigma
            if sigma > 0
            else float("inf")
        )
        if significance < threshold:
            continue
        lo_i, hi_i = int(left_bases[k]), int(right_bases[k]) + 1
        seg_x, seg_y = x[lo_i:hi_i], y[lo_i:hi_i] - base
        area = float(np.trapezoid(np.clip(seg_y, 0.0, None), seg_x)) if seg_x.size > 1 else 0.0
        peaks.append(
            PeakMeasurement(
                position=position,
                height=height,
                fwhm=fwhm,
                area=area,
                prominence=float(prominences[k]),
                snr=height / sigma if sigma > 0 else float("inf"),
                significance=float(significance),
            )
        )
    peaks.sort(key=lambda p: p.position)
    return peaks


#: Window length, in points, the default detection threshold was calibrated on.
CALIBRATION_POINTS = 320.0


def _trials_factor(n_points: int) -> float:
    """Look-elsewhere scaling of the detection threshold.

    The expected maximum of *n* independent Gaussian samples grows as
    ``sqrt(2 ln n)``, so a threshold that holds the false-alarm rate fixed
    must grow as ``sqrt(ln n)``. Normalised to the 320-point window the
    default was calibrated on.
    """
    if n_points <= 1:
        return 1.0
    return float(np.sqrt(np.log(n_points) / np.log(CALIBRATION_POINTS)))


def _parabolic_vertex(x: np.ndarray, y: np.ndarray, idx: int) -> tuple[float, float]:
    """Sub-sample peak position and height from a 3-point parabola.

    An instrument sampling at 2 cm⁻¹ quantises every reported band position
    to 2 cm⁻¹ unless this is done, and the shifts this package is asked to
    detect — doping-induced G shifts of 3–8 cm⁻¹ — are of that order.
    """
    if idx <= 0 or idx >= x.size - 1:
        return float(x[idx]), float(y[idx])
    y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-30:
        return float(x[idx]), float(y[idx])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    spacing = float(x[idx + 1] - x[idx - 1]) / 2.0
    return float(x[idx] + delta * spacing), float(y1 - 0.25 * (y0 - y2) * delta)


def measure_peak(
    spectrum: Spectrum,
    approximate_position: float,
    window: float = 60.0,
    baseline: str = "linear",
) -> Optional[PeakMeasurement]:
    """Measure one band around a guessed position, without fitting a model.

    A local background is taken as the straight line between the two window
    edges (``baseline="linear"``) or as zero (``baseline="none"``, correct
    when the spectrum was already baseline-corrected and the band is
    isolated). Height, FWHM and area are all referred to that line.

    Parameters
    ----------
    spectrum:
        The spectrum to measure.
    approximate_position:
        Where to look, in cm⁻¹.
    window:
        Half-width of the search window in cm⁻¹. Too wide and a neighbouring
        band becomes the local maximum; for the D/G region 60 cm⁻¹ keeps D
        and G apart but will still catch D' inside the G window.
    baseline:
        ``"linear"`` or ``"none"``.

    Returns
    -------
    PeakMeasurement or None
        ``None`` if the window holds fewer than five points.
    """
    lo, hi = approximate_position - window, approximate_position + window
    x, y = spectrum.region(lo, hi)
    if x.size < 5:
        return None

    if baseline == "linear":
        edge = max(1, x.size // 20)
        x0, y0 = float(np.mean(x[:edge])), float(np.mean(y[:edge]))
        x1, y1 = float(np.mean(x[-edge:])), float(np.mean(y[-edge:]))
        slope = (y1 - y0) / (x1 - x0) if abs(x1 - x0) > 1e-12 else 0.0
        local = y - (y0 + slope * (x - x0))
    elif baseline == "none":
        local = y.copy()
    else:
        raise ValueError(f"unknown local baseline {baseline!r}")

    idx = int(np.argmax(local))
    position, height = _parabolic_vertex(x, local, idx)
    if height <= 0:
        return None

    fwhm = _fwhm_from_data(x, local, idx, height)
    area = float(np.trapezoid(np.clip(local, 0.0, None), x))
    sigma = spectrum.noise_estimate()
    snr = height / sigma if sigma > 0 else float("inf")
    prominence = height - float(np.min(local))

    return PeakMeasurement(
        position=position,
        height=height,
        fwhm=fwhm,
        area=area,
        prominence=prominence,
        snr=snr,
    )


def _fwhm_from_data(
    x: np.ndarray, y: np.ndarray, idx: int, height: float
) -> Optional[float]:
    """Full width at half maximum by linear interpolation of the crossings.

    Returns ``None`` when the curve does not come back down to half height
    on both sides inside the window — the honest answer for a shoulder,
    which is exactly the situation (D' on the G band) where a fabricated
    width would mislead most.
    """
    half = height / 2.0

    left = None
    for i in range(idx, 0, -1):
        if y[i - 1] <= half <= y[i]:
            span = y[i] - y[i - 1]
            frac = (half - y[i - 1]) / span if abs(span) > 1e-30 else 0.0
            left = x[i - 1] + frac * (x[i] - x[i - 1])
            break

    right = None
    for i in range(idx, x.size - 1):
        if y[i + 1] <= half <= y[i]:
            span = y[i] - y[i + 1]
            frac = (y[i] - half) / span if abs(span) > 1e-30 else 0.0
            right = x[i] + frac * (x[i + 1] - x[i])
            break

    if left is None or right is None or right <= left:
        return None
    return float(right - left)


def strongest_in(
    peaks: Sequence[PeakMeasurement], low: float, high: float
) -> Optional[PeakMeasurement]:
    """The tallest measured peak inside a window, or ``None``."""
    inside = [p for p in peaks if low <= p.position <= high]
    if not inside:
        return None
    return max(inside, key=lambda p: p.height)


__all__ = [
    "CALIBRATION_POINTS",
    "PeakMeasurement",
    "find_peaks",
    "measure_peak",
    "strongest_in",
]
