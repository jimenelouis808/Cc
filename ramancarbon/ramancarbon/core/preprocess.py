"""Cosmic-ray removal, smoothing, resampling and normalisation.

The order matters and is not arbitrary:

1. **Despike.** A cosmic ray is a one-or-two-pixel spike that no amount of
   later fitting will survive; it must go before anything smooths it into a
   plausible-looking narrow band.
2. **Resample** onto a uniform axis, if the instrument did not already
   produce one. Savitzky–Golay assumes uniform spacing.
3. **Smooth**, gently or not at all. Smoothing narrows nothing and broadens
   everything: a Savitzky–Golay window wider than the narrowest real
   feature inflates FWHM and biases every deconvolution. The default window
   here is deliberately small.
4. **Baseline** (in :mod:`ramancarbon.core.baseline`).
5. **Normalise**, last, because normalisation by a band's height is
   meaningless until the background under that band is gone.

Every function returns a new :class:`~ramancarbon.core.spectrum.Spectrum`
with the step appended to ``history``; nothing mutates in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import medfilt, savgol_filter

from .spectrum import Spectrum

#: Below this ratio of noise σ to full intensity range, a spectrum is treated
#: as noise-free and despiking is skipped.
#:
#: Real spectra sit far above it: an intensity range of 1000 counts with a
#: noise σ of 3 gives 3e-3, and even a very high signal-to-noise measurement
#: stays near 1e-4. A *noiseless* curve gives ~1e-6, because there the
#: second-difference estimator is measuring band curvature rather than noise.
#: The two regimes are two orders of magnitude apart, so the boundary is not
#: delicate.
NOISE_FLOOR_FRACTION = 1e-5


def despike(
    spectrum: Spectrum,
    threshold: float = 8.0,
    kernel: int = 5,
    positive_only: bool = True,
) -> tuple[Spectrum, np.ndarray]:
    """Remove cosmic-ray spikes.

    A point is a spike if it stands far off the running median of its
    neighbourhood. The running median is the right reference here because
    it is exact on a straight line: the median of five monotonically
    increasing points *is* the middle one, so the steep flank of an intense
    G band leaves no residual and is never flagged. A cosmic ray, being one
    or two pixels wide, cannot survive a five-point median and stands out
    by many σ.

    This replaces the first-difference (Whitaker–Hayes) criterion that a
    previous version of this module used. That criterion scales with the
    *slope* of the data, and on a sharp, intense band — a 20 cm⁻¹ G band
    sampled at 1 cm⁻¹ — the flank slope exceeds the noise slope by more
    than the usual threshold of 7, so the flanks were being carved out and
    the band was left with notches in its sides.

    Parameters
    ----------
    spectrum:
        Input spectrum.
    threshold:
        How many noise σ a point must stand above the running median to
        count as a spike. 6–10 is the usual range. σ comes from
        :meth:`Spectrum.noise_estimate`, which is derived from the second
        difference and is therefore insensitive to band curvature.
    kernel:
        Median-filter length in points; forced odd, minimum 3. It sets the
        widest spike that can be removed: ``kernel // 2`` points. Raising it
        above 7 starts clipping the tops of genuinely sharp RBM peaks.
    positive_only:
        Only remove upward spikes. Cosmic rays add charge, they never
        remove it, so a downward outlier is a detector problem the user
        should see rather than something to paper over.

    Returns
    -------
    (Spectrum, numpy.ndarray)
        The cleaned spectrum and a boolean mask of the points that were
        replaced, so a caller can show the user exactly what was removed.

    Warnings
    --------
    A band only a few points wide is, to a median filter, indistinguishable
    from a spike. If the instrument sampled at 4 cm⁻¹ or coarser, or the
    sample shows very sharp RBMs, drop ``kernel`` to 3 or turn despiking
    off; otherwise the apex of the narrowest band can be clipped.

    Run this **before** smoothing. Smoothing spreads a cosmic ray over the
    whole smoothing window, turning a removable one-pixel artefact into a
    plausible-looking narrow band that no later step can identify.
    """
    k = int(kernel)
    if k % 2 == 0:
        k += 1
    k = max(k, 3)
    y = spectrum.intensity
    n = y.size
    if n < 2 * k + 1:
        return spectrum.copy(), np.zeros(n, dtype=bool)

    # Scale against the second-difference noise estimate, not against the
    # spread of the median residual itself. That residual carries the
    # curvature of every band top, so normalising by its own MAD makes the
    # threshold depend on how peaky the spectrum is and clips sharp apices.
    sigma = spectrum.noise_estimate()
    scale = float(np.ptp(y))
    if sigma <= NOISE_FLOOR_FRACTION * max(scale, 1e-30):
        # No measurable noise means no cosmic rays to find, and no scale
        # against which to call a point an outlier. On such data the
        # estimator is reporting band curvature rather than noise, and
        # thresholding on it clips the apex of the sharpest band. Do
        # nothing, and record that nothing was done.
        return (
            spectrum.with_intensity(y, "despike(sin ruido medible, sin cambios)"),
            np.zeros(n, dtype=bool),
        )

    smoothed = medfilt(y, kernel_size=k)
    residual = y - smoothed
    flagged = residual > threshold * sigma
    if not positive_only:
        flagged |= residual < -threshold * sigma

    cleaned = y.copy()
    cleaned[flagged] = smoothed[flagged]
    # The history entry is written even when nothing was removed, so that a
    # provenance record distinguishes "despiking found nothing" from
    # "despiking was switched off".
    label = f"despike(threshold={threshold:g}, kernel={k}, n={int(flagged.sum())})"
    return spectrum.with_intensity(cleaned, label), flagged


def resample(spectrum: Spectrum, step: Optional[float] = None) -> Spectrum:
    """Interpolate onto a uniform Raman-shift axis.

    Parameters
    ----------
    spectrum:
        Input spectrum, possibly unevenly sampled.
    step:
        Target spacing in cm⁻¹. Defaults to the input's median spacing,
        which resamples without changing the effective resolution.

    Returns
    -------
    Spectrum
        Uniformly sampled over the original range.
    """
    target = float(step) if step else spectrum.step
    if target <= 0:
        raise ValueError("resampling step must be positive")
    lo, hi = spectrum.range
    n = int(np.floor((hi - lo) / target)) + 1
    if n < 2:
        raise ValueError(f"step {target:g} cm⁻¹ is wider than the spectrum")
    axis = lo + target * np.arange(n)
    out = spectrum.copy()
    out.intensity = spectrum.interpolate_at(axis)
    out.shift = axis
    out.history.append(f"resample(step={target:g})")
    return out


def smooth(spectrum: Spectrum, window: int = 9, order: int = 3) -> Spectrum:
    """Savitzky–Golay smoothing.

    Parameters
    ----------
    spectrum:
        Input spectrum, ideally uniformly sampled (see :func:`resample`).
    window:
        Window length in points; forced odd and to at least ``order + 2``.
        As a rule of thumb keep it below one third of the narrowest band's
        FWHM in points — for a 12 cm⁻¹ RBM sampled at 1 cm⁻¹, that is 4–5
        points, not 15.
    order:
        Polynomial order. 2 or 3 preserves peak height far better than 1.

    Returns
    -------
    Spectrum
        Smoothed copy.

    Warnings
    --------
    Smoothing before a deconvolution correlates neighbouring residuals,
    which makes the fit's own error bars optimistic. This package's fitter
    reports that caveat when it sees ``smooth`` in the history.
    """
    if order < 1:
        raise ValueError("Savitzky-Golay order must be >= 1")
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = max(w, order + 2 + (order % 2))
    if w % 2 == 0:
        w += 1
    if w >= spectrum.shift.size:
        raise ValueError(
            f"smoothing window ({w} points) is not shorter than the spectrum "
            f"({spectrum.shift.size} points)"
        )
    smoothed = savgol_filter(spectrum.intensity, window_length=w, polyorder=order)
    return spectrum.with_intensity(smoothed, f"smooth(savgol, window={w}, order={order})")


def normalise(
    spectrum: Spectrum,
    method: str = "max",
    window: Optional[tuple[float, float]] = None,
) -> Spectrum:
    """Scale the intensity to a common reference.

    Parameters
    ----------
    spectrum:
        Input spectrum, normally already baseline-corrected.
    method:
        ``"max"``
            Divide by the global maximum (or by the maximum inside
            ``window``).
        ``"g"``
            Divide by the maximum in 1500–1650 cm⁻¹, the G band. The
            convention used when overlaying carbon spectra, because the G
            band is present in every sp² carbon and is the one band whose
            intensity is not defect-controlled.
        ``"area"``
            Divide by the trapezoidal area (over ``window`` if given).
        ``"minmax"``
            Map to [0, 1].
        ``"0-100"``
            Map to [0, 100]. The convention for overlaying spectra on a
            percentage axis; identical to ``minmax`` up to the factor of
            100.

            Unlike every other option here, min-max scaling subtracts an
            **offset** as well as applying a factor, and an offset does not
            cancel in a ratio. If the background has already been removed
            the minimum is near zero and the effect is negligible; on a
            spectrum that still carries a background it is not, and I_D/I_G
            measured after normalising will differ from the same ratio
            measured before. Baseline-correct first.
    window:
        Optional ``(low, high)`` in cm⁻¹ restricting the reference.

    Returns
    -------
    Spectrum
        Scaled copy.

    Notes
    -----
    Scaling never changes an intensity ratio, so ``"max"``, ``"g"`` and
    ``"area"`` leave I_D/I_G untouched. ``"minmax"`` and ``"0-100"`` are
    the exception: they subtract the minimum, and that offset does not
    cancel in a ratio. On baseline-corrected data the minimum is ~0 and the
    difference is fractions of a percent; on uncorrected data it can be
    much larger, and the returned spectrum's history says so.
    """
    y = spectrum.intensity
    key = method.lower()
    if key == "g":
        lo, hi = window or (1500.0, 1650.0)
        peak = spectrum.max_in(lo, hi)
        if peak is None:
            raise ValueError(f"no data in the G window {lo:g}–{hi:g} cm⁻¹")
        scale = peak[1]
        label = f"normalise(G, {lo:g}-{hi:g})"
    elif key == "max":
        if window:
            peak = spectrum.max_in(*window)
            if peak is None:
                raise ValueError(f"no data in {window[0]:g}–{window[1]:g} cm⁻¹")
            scale = peak[1]
        else:
            scale = float(np.max(y))
        label = "normalise(max)"
    elif key == "area":
        lo, hi = window or spectrum.range
        scale = spectrum.area_in(lo, hi)
        label = f"normalise(area, {lo:g}-{hi:g})"
    elif key in {"minmax", "0-100", "0100", "percent"}:
        top = 100.0 if key != "minmax" else 1.0
        lo_v, hi_v = float(np.min(y)), float(np.max(y))
        if hi_v - lo_v <= 0:
            raise ValueError("cannot min-max normalise a flat spectrum")
        label = "normalise(0-100)" if top == 100.0 else "normalise(minmax)"
        offset_fraction = abs(lo_v) / (hi_v - lo_v)
        if offset_fraction > 0.02:
            label += (
                f" [AVISO: resta un desplazamiento del {offset_fraction * 100:.0f} % "
                "del rango; los cocientes cambian. Resta la línea base antes]"
            )
        return spectrum.with_intensity(top * (y - lo_v) / (hi_v - lo_v), label)
    else:
        raise ValueError(f"unknown normalisation {method!r}")

    if not np.isfinite(scale) or abs(scale) < 1e-30:
        raise ValueError(f"normalisation reference is zero or non-finite ({scale!r})")
    return spectrum.with_intensity(y / scale, label)


@dataclass
class AutoSettings:
    """Preprocessing parameters chosen from the data, and why.

    Every field is a suggestion the user can override; ``reasons`` carries
    one sentence per decision so the GUI can show what it did and the user
    can disagree with a specific choice rather than with the whole thing.
    """

    despike: bool
    smooth_window: int
    baseline_method: str
    baseline_lam: float
    reasons: dict[str, str] = field(default_factory=dict)

    def to_kwargs(self) -> dict:
        """The arguments :func:`preprocess` needs."""
        return {
            "do_despike": self.despike,
            "smooth_window": self.smooth_window,
            "baseline_method": self.baseline_method,
            "baseline_kwargs": {"lam": self.baseline_lam},
        }

    def summary(self) -> str:
        """Multi-line explanation of every choice."""
        lines = [
            f"despiking      : {'sí' if self.despike else 'no'}",
            f"suavizado      : {self.smooth_window or 'ninguno'} puntos",
            f"línea base     : {self.baseline_method}, λ = {self.baseline_lam:.0e}",
            "",
        ]
        lines.extend(f"• {text}" for text in self.reasons.values())
        return "\n".join(lines)


def auto_settings(
    spectrum: Spectrum,
    target_snr: float = 40.0,
    baseline_method: str = "asls",
) -> AutoSettings:
    """Pick preprocessing parameters from the spectrum itself.

    Three decisions, each made from a measurement rather than a default:

    **Smoothing.** Smoothing trades resolution for signal-to-noise, so it
    is only worth doing when the noise is actually limiting. The strongest
    band's signal-to-noise ratio is measured; if it already exceeds
    ``target_snr`` nothing is smoothed. If it does not, the window is set
    to one third of the *narrowest* real feature in the spectrum — because
    a Savitzky–Golay window wider than that broadens the feature, and an
    inflated FWHM propagates into every width-based conclusion. The window
    is capped so it can never exceed a third of the narrowest band even if
    the noise would justify more; below that cap, noisy data simply stays
    noisy and the fit's error bars say so honestly.

    **Baseline stiffness.** Delegated to
    :func:`~ramancarbon.core.baseline.auto_lambda`, which returns the
    stiffest baseline that still tracks the background within the noise.

    **Despiking.** On unless the spectrum has no measurable noise, in
    which case there are no cosmic rays to find.

    Parameters
    ----------
    spectrum:
        Raw spectrum.
    target_snr:
        Signal-to-noise above which smoothing is judged unnecessary. 40 is
        comfortable for peak fitting; raising it makes the routine smooth
        more eagerly.
    baseline_method:
        ``"asls"`` (default) or ``"arpls"``. See
        :mod:`ramancarbon.core.baseline` for why asLS is the default
        despite arPLS's reputation for fluorescence.

    Returns
    -------
    AutoSettings
    """
    from .baseline import auto_lambda  # local import: avoids a cycle
    from .peaks import find_peaks

    reasons: dict[str, str] = {}
    sigma = spectrum.noise_estimate()
    scale = float(np.ptp(spectrum.intensity))

    despike_on = sigma > NOISE_FLOOR_FRACTION * max(scale, 1e-30)
    reasons["despike"] = (
        "se eliminan rayos cósmicos"
        if despike_on
        else "no hay ruido medible, así que no hay rayos cósmicos que quitar"
    )

    # Work on a roughly background-free copy so the peak search and the SNR
    # are not dominated by fluorescence.
    probe = spectrum
    try:
        from .baseline import arpls_baseline

        probe = spectrum.with_intensity(
            spectrum.intensity - arpls_baseline(spectrum.intensity, lam=1e7), "probe"
        )
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - degenerate input
        pass

    peaks = find_peaks(probe)
    strongest = max((p.height for p in peaks), default=0.0)
    snr = strongest / sigma if sigma > 0 else float("inf")

    window = 0
    if not peaks:
        reasons["smooth"] = (
            "no se detectan bandas; no se suaviza, porque suavizar ruido no "
            "crea señal"
        )
    elif snr >= target_snr:
        reasons["smooth"] = (
            f"la banda más intensa tiene SNR ≈ {snr:.0f}, por encima de "
            f"{target_snr:.0f}: suavizar solo ensancharía las bandas sin ganar nada"
        )
    else:
        narrowest = min(p.fwhm for p in peaks if p.fwhm)
        step = max(spectrum.step, 1e-9)
        limit = max(5, int(narrowest / step / 3.0))
        wanted = int(np.clip(round((target_snr / max(snr, 1e-9)) ** 2), 5, 31))
        window = int(min(wanted, limit))
        if window % 2 == 0:
            window += 1
        reasons["smooth"] = (
            f"SNR ≈ {snr:.0f} por debajo de {target_snr:.0f}: se suaviza con "
            f"{window} puntos, un tercio de la banda más estrecha "
            f"({narrowest:.0f} cm⁻¹) para no ensancharla"
        )
        if wanted > limit:
            reasons["smooth"] += (
                f". El ruido justificaría {wanted} puntos, pero eso deformaría "
                "esa banda, así que se deja más ruidoso a propósito"
            )

    lam, lam_reason = auto_lambda(spectrum, method=baseline_method)
    reasons["baseline"] = lam_reason

    return AutoSettings(
        despike=despike_on,
        smooth_window=window,
        baseline_method=baseline_method,
        baseline_lam=lam,
        reasons=reasons,
    )


def preprocess(
    spectrum: Spectrum,
    do_despike: bool = True,
    do_resample: bool = False,
    smooth_window: int = 0,
    baseline_method: Optional[str] = "asls",
    baseline_kwargs: Optional[dict] = None,
    normalise_method: Optional[str] = None,
    crop: Optional[tuple[float, float]] = None,
) -> tuple[Spectrum, dict]:
    """Run the standard pipeline in the order the module docstring defends.

    Every stage is optional, but they always run in the same sequence.
    Returns the processed spectrum plus a diagnostics dictionary holding
    the spike mask and the baseline, so the GUI can draw both over the raw
    data.

    Parameters
    ----------
    spectrum:
        Raw input.
    do_despike:
        Run :func:`despike`.
    do_resample:
        Force a uniform axis. Automatically enabled when smoothing is
        requested on a non-uniform axis.
    smooth_window:
        Savitzky–Golay window in points; 0 disables smoothing.
    baseline_method:
        Passed to :func:`~ramancarbon.core.baseline.subtract_baseline`;
        ``None`` skips background removal.
    baseline_kwargs:
        Extra parameters for the baseline estimator.
    normalise_method:
        Passed to :func:`normalise`; ``None`` skips it.
    crop:
        Optional ``(low, high)`` applied first, before everything else.

    Returns
    -------
    (Spectrum, dict)
        Processed spectrum, and ``{"spikes": mask|None, "baseline": array|None,
        "baseline_x": array|None}``.
    """
    from .baseline import subtract_baseline  # local import: avoids a cycle

    diagnostics: dict = {"spikes": None, "baseline": None, "baseline_x": None}
    work = spectrum.crop(*crop) if crop else spectrum.copy()

    if do_despike:
        work, mask = despike(work)
        diagnostics["spikes"] = mask

    if do_resample or (smooth_window and not work.is_uniform):
        work = resample(work)
        diagnostics["spikes"] = None  # indices no longer refer to this axis

    if smooth_window and smooth_window > 2:
        work = smooth(work, window=smooth_window)

    if baseline_method:
        work, base = subtract_baseline(work, method=baseline_method, **(baseline_kwargs or {}))
        diagnostics["baseline"] = base
        diagnostics["baseline_x"] = work.shift.copy()

    if normalise_method:
        work = normalise(work, method=normalise_method)

    return work, diagnostics


__all__ = [
    "AutoSettings",
    "NOISE_FLOOR_FRACTION",
    "auto_settings",
    "despike",
    "normalise",
    "preprocess",
    "resample",
    "smooth",
]
