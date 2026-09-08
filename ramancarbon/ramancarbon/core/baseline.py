"""Background removal.

Carbon Raman spectra ride on a background that is rarely flat: fluorescence
from the sample or its dispersant, a rising thermal tail from a black
powder, the wing of the Rayleigh line under the RBM region. Every intensity
ratio this package reports — I_D/I_G above all — is a ratio of numbers that
still contain that background unless it is removed first, and a sloping
background inflates whichever band sits higher on the slope.

Three estimators are offered, in increasing order of how much they assume:

``asls``
    Asymmetric least squares (Eilers & Boelens 2005). Fits a smooth curve
    that is penalised much harder for going above the data than below it,
    so it slides under the peaks. **This is the default**, on evidence: see
    the note below.
``arpls``
    Asymmetrically **reweighted** penalized least squares (Baek et al.
    2015). Same idea, but the asymmetry is re-derived at each iteration
    from the statistics of the points lying *below* the current baseline
    rather than fixed by the user, so there is nothing to tune but the
    stiffness.
``polynomial``
    Iterative polynomial fit with peak rejection (Lieber & Mahadevan-Jansen
    2003). Cheap and stable, but a low order cannot follow a curved
    fluorescence tail and a high order starts eating the D band.
``rubberband``
    The lower convex hull. Assumption-free in shape but it only touches the
    data at a few points, so it under-subtracts wherever the true baseline
    is concave — which in the 1000–1800 cm⁻¹ window is most of it.

All of them return the *baseline*, not the corrected spectrum; the caller
subtracts. Keeping them separate lets the GUI draw the proposed baseline
over the raw data before anything is committed.

**Which of the two to use.** arPLS is often recommended for fluorescence,
and this package originally made it the default on that reputation. Measured
against synthetic carbon spectra with a known background — flat, linear,
single and double exponential, each at the stiffness :func:`auto_lambda`
would choose — asLS came out with the lower background error in every case,
and arPLS carried a systematic offset of a few noise units (it deliberately
settles slightly below the data, which is what makes its reweighting
stable). So asLS is the default here. arPLS remains available because it
needs no asymmetry parameter, which is useful when batch-processing spectra
whose backgrounds differ a lot.

**What both get right.** Even where the background error reached 140 counts
on a 3000-count fluorescence tail, the recovered I_D/I_G was within 3 % of
truth, for both methods. A smooth background error is nearly constant across
the 1300–1600 cm⁻¹ window and largely cancels in a ratio taken inside it.
Absolute areas are much more sensitive to the baseline than ratios are —
which is the argument for reporting ratios rather than areas whenever the
background is strong.

The stiffness ``lam`` is the one parameter that always matters, and
:func:`auto_lambda` picks it from the data: it returns the **stiffest**
baseline that still tracks the background to within the noise. Stiffest,
not best-fitting, because the failure that costs you a number is a baseline
soft enough to curl up into the D–G valley and shave intensity off both
bands.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .spectrum import Spectrum


def _second_difference_penalty(n: int):
    """``DᵀD`` for the second-difference operator, as a sparse matrix.

    Shared by every penalized-least-squares baseline here, so they all
    mean the same thing by "stiffness" and a ``lam`` tuned for one carries
    over to the other.
    """
    d = sparse.diags(
        [np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)],
        offsets=[0, 1, 2],
        shape=(n - 2, n),
        format="csc",
    )
    return (d.T @ d).tocsc()


def asls_baseline(
    y: Sequence[float],
    lam: float = 1e7,
    p: float = 0.001,
    max_iter: int = 30,
    tol: float = 1e-4,
) -> np.ndarray:
    """Asymmetric least squares baseline.

    Minimises ``Σ w_i (y_i − z_i)² + λ Σ (Δ²z_i)²`` where points above the
    current baseline get weight ``p`` and points below get ``1 − p``. Peaks,
    being above, are almost ignored; the baseline settles on the valleys.

    Parameters
    ----------
    y:
        Intensities, evenly or unevenly sampled. The smoothness penalty is
        expressed in *points*, not in cm⁻¹, so a strongly non-uniform axis
        makes ``lam`` mean different things in different regions; resample
        first if that matters.
    lam:
        Smoothness. Larger is stiffer. 1e5–1e8 is the useful range for a
        typical 1 cm⁻¹-per-point carbon spectrum. The default 1e7 was
        chosen against synthetic spectra with a known exponential
        fluorescence tail: softer values let the baseline curl up into the
        D–G valley and shave ~15 % off the G height, which would propagate
        straight into I_D/I_G.
    p:
        Asymmetry, in (0, 1). Smaller pushes the baseline further down.
        1e-4–0.05 is usual; 0.5 would be an ordinary smoother. Carbon
        spectra are mostly peak between 1200 and 1700 cm⁻¹, so they need a
        smaller ``p`` than the 0.01 usually quoted for sparse-peak
        spectroscopies.
    max_iter:
        Iteration cap for the reweighting.
    tol:
        Stop when the relative change in the weights falls below this.

    Returns
    -------
    numpy.ndarray
        The baseline, same length as ``y``.

    Raises
    ------
    ValueError
        If ``p`` is not strictly inside (0, 1) or ``lam`` is not positive.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"asymmetry p must be in (0, 1), got {p!r}")
    if lam <= 0:
        raise ValueError(f"smoothness lam must be positive, got {lam!r}")
    y_arr = np.asarray(y, dtype=float).ravel()
    n = y_arr.size
    if n < 5:
        return np.full(n, float(np.min(y_arr)))

    penalty = lam * _second_difference_penalty(n)
    w = np.ones(n)
    z = y_arr.copy()
    for _ in range(max_iter):
        w_mat = sparse.diags(w, 0, shape=(n, n), format="csc")
        z = spsolve((w_mat + penalty).tocsc(), w * y_arr)
        w_new = np.where(y_arr > z, p, 1.0 - p)
        if np.linalg.norm(w_new - w) / max(np.linalg.norm(w), 1e-12) < tol:
            w = w_new
            break
        w = w_new
    return np.asarray(z, dtype=float)


def arpls_baseline(
    y: Sequence[float],
    lam: float = 1e6,
    ratio: float = 1e-3,
    max_iter: int = 50,
) -> np.ndarray:
    """Asymmetrically reweighted penalized least squares baseline.

    The difference from :func:`asls_baseline` is where the asymmetry comes
    from. asLS applies a fixed weight ``p`` to every point above the
    current baseline. arPLS instead looks at the points that fell *below*
    it — which, once the baseline is roughly right, are pure background
    plus noise — measures their mean ``m`` and standard deviation ``s``,
    and gives each point a logistic weight::

        w_i = 1 / (1 + exp(2 (d_i - (2s - m)) / s))

    A point a few noise units above the background gets essentially zero
    weight; a point within the noise keeps weight near one. So the
    threshold between "signal" and "background" is re-derived from the data
    at every iteration instead of being asserted once by the user.

    The trade-off, measured rather than assumed: this makes arPLS
    parameter-free in its asymmetry, but it also biases the baseline a few
    noise units away from the data, because the logistic threshold sits at
    ``2s - m`` rather than at zero. On the synthetic backgrounds used to
    calibrate this module, asLS reached a lower error in every case. Prefer
    :func:`asls_baseline` unless you specifically want to avoid choosing
    ``p``.

    Parameters
    ----------
    y:
        Intensities. The smoothness penalty is expressed in *points*, so
        resample a strongly non-uniform axis first.
    lam:
        Smoothness. Larger is stiffer. See :func:`auto_lambda` to pick it
        from the data.
    ratio:
        Convergence tolerance on the relative change of the weight vector.
    max_iter:
        Iteration cap. Convergence is usually reached in 10–20.

    Returns
    -------
    numpy.ndarray
        The baseline, same length as ``y``.

    Raises
    ------
    ValueError
        If ``lam`` is not positive.

    References
    ----------
    Baek, Park, Ahn & Choo, *Analyst* **140** (2015) 250.
    """
    if lam <= 0:
        raise ValueError(f"smoothness lam must be positive, got {lam!r}")
    y_arr = np.asarray(y, dtype=float).ravel()
    n = y_arr.size
    if n < 5:
        return np.full(n, float(np.min(y_arr)))

    penalty = lam * _second_difference_penalty(n)
    w = np.ones(n)
    z = y_arr.copy()
    for _ in range(max_iter):
        w_mat = sparse.diags(w, 0, shape=(n, n), format="csc")
        z = spsolve((w_mat + penalty).tocsc(), w * y_arr)
        d = y_arr - z
        negative = d[d < 0]
        if negative.size < 2:
            break
        mean = float(negative.mean())
        std = float(negative.std())
        if std <= 0:
            break
        # Logistic reweighting. The exponent is clipped because a strongly
        # fluorescent spectrum can put a peak hundreds of sigma above the
        # background, and exp() of that overflows to a warning and a nan.
        exponent = np.clip(2.0 * (d - (2.0 * std - mean)) / std, -50.0, 50.0)
        w_new = 1.0 / (1.0 + np.exp(exponent))
        if np.linalg.norm(w - w_new) / max(np.linalg.norm(w), 1e-12) < ratio:
            w = w_new
            break
        w = w_new
    return np.asarray(z, dtype=float)


def polynomial_baseline(
    x: Sequence[float],
    y: Sequence[float],
    order: int = 3,
    max_iter: int = 40,
    tol: float = 1e-3,
) -> np.ndarray:
    """Iterative polynomial baseline with peak clipping.

    Fits a polynomial, then replaces every point that lies above the fit by
    the fit itself and refits. Peaks are progressively clipped away and the
    polynomial converges onto the background.

    Parameters
    ----------
    x, y:
        Shift axis and intensities.
    order:
        Polynomial degree. 1–3 for a gently sloping background; going above
        5 over a wide window lets the polynomial follow the D–G envelope
        and eat real signal.
    max_iter, tol:
        Convergence controls; iteration stops when the fitted curve changes
        by less than ``tol`` times its own RMS.

    Returns
    -------
    numpy.ndarray
        The baseline.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if order < 0:
        raise ValueError("polynomial order must be >= 0")
    if x_arr.size <= order + 1:
        return np.full(y_arr.size, float(np.min(y_arr)))

    # Centre and scale the abscissa; a raw 100–3200 cm^-1 axis makes the
    # Vandermonde matrix of a cubic badly conditioned.
    xc = (x_arr - x_arr.mean()) / max(x_arr.std(), 1e-12)
    work = y_arr.copy()
    previous = None
    for _ in range(max_iter):
        coeffs = np.polyfit(xc, work, order)
        fit = np.polyval(coeffs, xc)
        work = np.minimum(work, fit)
        if previous is not None:
            change = np.sqrt(np.mean((fit - previous) ** 2))
            scale = max(np.sqrt(np.mean(fit**2)), 1e-12)
            if change / scale < tol:
                previous = fit
                break
        previous = fit
    return np.asarray(previous, dtype=float)


def rubberband_baseline(x: Sequence[float], y: Sequence[float]) -> np.ndarray:
    """Lower convex hull ("rubber band") baseline.

    Imagine stretching a rubber band under the spectrum: it touches the
    lowest points and runs straight between them. Nothing is assumed about
    the background's functional form, which is its virtue; the cost is that
    it is piecewise linear and always sits at or below the true background,
    so it under-subtracts in concave regions.

    Returns
    -------
    numpy.ndarray
        The baseline.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    n = x_arr.size
    if n < 3:
        return np.full(n, float(np.min(y_arr)))

    # Andrew's monotone chain, lower hull only. The axis is already sorted.
    hull: list[int] = []
    for i in range(n):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = (x_arr[b] - x_arr[a]) * (y_arr[i] - y_arr[a]) - (
                y_arr[b] - y_arr[a]
            ) * (x_arr[i] - x_arr[a])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    idx = np.array(hull, dtype=int)
    return np.interp(x_arr, x_arr[idx], y_arr[idx])


#: How many band widths the baseline's cutoff period is set to. Calibrated
#: against synthetic spectra with a known fluorescence tail: below ~3 the
#: baseline starts following the bands, above ~8 it stops following the
#: background. Between 4 and 6 the recovered I_D/I_G is within a few percent
#: of truth, and 5 is the middle of that plateau.
CUTOFF_BAND_WIDTHS = 5.0

#: Bounds on the automatically chosen stiffness.
#:
#: The upper end is where the banded solve starts to lose conditioning:
#: measured on a broad synthetic band, results are stable to 1e12 and the
#: baseline range jumps at 1e13. Very broad-band material (amorphous carbon,
#: heavily disordered fibres with 300 cm⁻¹ bands) genuinely asks for
#: stiffnesses above 1e10, so the ceiling is not merely decorative.
LAMBDA_LIMITS = (1e4, 1e12)

#: Floor on the "widest band" the cutoff is derived from, in cm⁻¹.
#:
#: On a noisy spectrum the peak finder may only recover the sharp bands and
#: miss the broad 2D envelope, which would make the cutoff too soft and let
#: the baseline eat band intensity. No carbon spectrum's broadest feature is
#: narrower than this — the G band alone reaches 80–100 cm⁻¹ in multi-walled
#: and fibre material — so flooring it errs towards a stiffer baseline, which
#: is the recoverable direction.
MIN_WIDEST_BAND_CM = 80.0


def lambda_for_cutoff(period_points: float) -> float:
    """Stiffness whose half-power cutoff is at a given period, in points.

    The Whittaker smoother behind :func:`asls_baseline` and
    :func:`arpls_baseline` has the transfer function

        ``H(ω) = 1 / (1 + 16 λ sin⁴(ω/2))``

    so it passes structure slower than, and rejects structure faster than,
    the period at which ``16 λ sin⁴(ω/2) = 1``. For the long periods that
    matter here that gives ``P ≈ π (16 λ)^(1/4)``, and inverting it turns
    "which features should the baseline follow" — a question about the
    sample — into a value of λ.

    Parameters
    ----------
    period_points:
        Desired half-power period, in data points.

    Returns
    -------
    float
        The corresponding ``lam``.
    """
    if period_points <= 0:
        raise ValueError("cutoff period must be positive")
    return float((period_points / np.pi) ** 4 / 16.0)


def cutoff_for_lambda(lam: float) -> float:
    """Inverse of :func:`lambda_for_cutoff`: the cutoff period, in points."""
    if lam <= 0:
        raise ValueError("lam must be positive")
    return float(np.pi * (16.0 * lam) ** 0.25)


def background_mask(
    y: Sequence[float],
    percentile: float = 35.0,
    smooth_points: int = 15,
    detrend: bool = True,
) -> np.ndarray:
    """Boolean mask of the points that carry no band, only background.

    Ranking raw intensity does not work on a fluorescent spectrum, and the
    way it fails is counterproductive: the lowest points are then simply
    wherever the fluorescence is weakest, which is the *far* end of the
    spectrum — so the mask systematically excludes the steep region near
    the laser line, the one place a baseline is most likely to go wrong.

    So the signal is detrended first, with a soft baseline that follows the
    background closely and leaves the bands standing, and the lowest
    ``percentile`` of *that* residual is taken. Band-free points are then
    found at any background level.

    Parameters
    ----------
    y:
        Intensities.
    percentile:
        Fraction of the spectrum treated as band-free. 35 % suits a carbon
        spectrum, where the D–G complex and the 2D region take up most of
        the rest.
    smooth_points:
        Width of the moving average used to suppress noise before ranking.
    detrend:
        Remove a soft background before ranking. Turn it off only when the
        input is known to be background-free already.

    Returns
    -------
    numpy.ndarray
        Boolean mask, same length as ``y``.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    n = y_arr.size
    if n < 3 * smooth_points:
        return np.ones(n, dtype=bool)
    work = y_arr
    if detrend:
        try:
            work = y_arr - arpls_baseline(y_arr, lam=lambda_for_cutoff(300.0))
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
            work = y_arr
    kernel = np.ones(int(smooth_points)) / float(smooth_points)
    smoothed = np.convolve(work, kernel, mode="same")
    return smoothed <= np.percentile(smoothed, percentile)


def auto_lambda(
    spectrum: "Spectrum",
    method: str = "arpls",
    band_widths: float = CUTOFF_BAND_WIDTHS,
    widest_band_cm: Optional[float] = None,
) -> tuple[float, str]:
    """Choose the baseline stiffness from the data.

    The rule is a statement about what the baseline is allowed to follow:
    **the cutoff period is set to a few times the widest band in the
    spectrum**, so anything as narrow as a band is rejected and anything
    slower passes as background. :func:`lambda_for_cutoff` turns that into
    a number.

    This replaced a grid search that scored each λ by how well its baseline
    matched the background-only points. That criterion does not work, and
    it is worth saying why, because it is the obvious thing to try: the
    residual over background points is dominated by noise and barely
    responds to λ, while the *systematic* part of it is minimised by the
    softest baseline on the grid — the one that follows the peak flanks and
    eats them. Both readings of the residual point the wrong way. The
    cutoff rule does not use the residual at all.

    Note that the answer depends on the **sampling step**, because the
    cutoff is a period in points. The same sample measured at 1 and at
    4 cm⁻¹ per point needs stiffnesses differing by a factor of 256, which
    is exactly the trap in copying a λ from a paper.

    Parameters
    ----------
    spectrum:
        The spectrum to choose a stiffness for.
    method:
        ``"arpls"`` or ``"asls"``; used only for the diagnostic check.
    band_widths:
        How many band widths the cutoff period is set to.
    widest_band_cm:
        Widest band FWHM in cm⁻¹, if you already know it. When omitted it
        is measured with the peak finder, falling back to 120 cm⁻¹ — a
        typical 2D band — if no peaks are found.

    Returns
    -------
    (float, str)
        The chosen stiffness and a sentence explaining it, which the GUI
        shows beside the control it has just set.
    """
    from .peaks import find_peaks  # local import: avoids a cycle

    step = max(spectrum.step, 1e-9)
    widest = widest_band_cm
    origin = "valor por defecto"
    if widest is None:
        try:
            probe = spectrum.with_intensity(
                spectrum.intensity - arpls_baseline(spectrum.intensity, lam=1e7),
                "probe",
            )
            widths = [p.fwhm for p in find_peaks(probe) if p.fwhm]
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
            widths = []
        if widths:
            widest = max(float(max(widths)), MIN_WIDEST_BAND_CM)
            origin = "medida en el propio espectro"
        else:
            widest = 120.0

    period = band_widths * widest / step
    lam = float(np.clip(lambda_for_cutoff(period), *LAMBDA_LIMITS))

    reason = (
        f"λ = {lam:.0e}: la línea base deja de seguir cualquier estructura más "
        f"estrecha que {band_widths:g} × {widest:.0f} cm⁻¹ = {period * step:.0f} cm⁻¹ "
        f"(la banda más ancha, {origin}), así que pasa por debajo de las bandas "
        f"pero sigue la fluorescencia"
    )

    fast, detail = _background_too_fast(spectrum, lam, method)
    if fast:
        reason += ". " + detail
    return lam, reason


def _background_too_fast(
    spectrum: "Spectrum", lam: float, method: str
) -> tuple[bool, str]:
    """Whether the background varies as fast as the bands themselves.

    Compares the chosen baseline against a much softer one over the
    background-only points. If they disagree by more than a few noise σ,
    the background has structure the chosen cutoff is rejecting — and no
    single stiffness can then separate background from signal, which the
    user needs to know rather than discover later in a wrong ratio.
    """
    estimator = arpls_baseline if method == "arpls" else asls_baseline
    y = spectrum.intensity
    sigma = spectrum.noise_estimate()
    if sigma <= 0:
        return False, ""
    mask = background_mask(y)
    if mask.sum() < 10:
        return False, ""
    try:
        stiff = estimator(y, lam=lam)
        soft = estimator(y, lam=max(lam / 100.0, LAMBDA_LIMITS[0]))
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
        return False, ""
    disagreement = float(np.sqrt(np.mean((stiff[mask] - soft[mask]) ** 2))) / sigma
    if disagreement < 4.0:
        return False, ""
    return True, (
        f"AVISO: el fondo varía casi tan rápido como las bandas "
        f"(discrepancia de {disagreement:.0f}σ entre líneas base rígida y "
        "blanda). Ninguna línea base puede separarlos limpiamente aquí; "
        "prueba a recortar el extremo de bajo desplazamiento, a medir con "
        "otro láser, o a fotoblanquear la muestra antes de medir"
    )


def snip_baseline(
    y: Sequence[float],
    iterations: Optional[int] = None,
    decreasing: bool = True,
    smooth_orders: int = 0,
) -> np.ndarray:
    """SNIP: statistics-sensitive non-linear iterative peak clipping.

    Each point is repeatedly replaced by the smaller of itself and the
    mean of the two points ``p`` channels away, with ``p`` running out to
    ``iterations``. Anything narrower than that window is clipped away and
    what is left is the background.

    The one parameter is a **width in channels**, which is why this method
    is worth having alongside asLS: it is stated in the same units the
    user already thinks in — "nothing I care about is wider than 80
    channels" — where a Whittaker λ has to be derived from the sampling
    step. It is also fast enough to run on every pixel of a map, which
    asLS is not: SNIP is a few array operations per iteration and no
    linear system at all.

    The clipping is done on the doubly-logarithmic (LLS) transform of the
    data, which is what makes it work on Poisson counts: on that scale the
    noise is roughly constant, so the same window is right at the top of a
    peak and in the background beside it.

    Parameters
    ----------
    y:
        Intensities. Must be finite; negative values are lifted before the
        transform and put back afterwards.
    iterations:
        Half-width in channels of the widest feature to remove. When
        ``None``, a twentieth of the spectrum, which is the usual default
        and is only ever a starting point.
    decreasing:
        Run the window from wide to narrow instead of narrow to wide. It
        preserves peak shape better and costs nothing; on by default.
    smooth_orders:
        Average over ``2·smooth_orders+1`` channels inside each iteration.
        Helps on very noisy data at the price of rounding sharp steps.

    Returns
    -------
    numpy.ndarray
        The baseline, on the input's own axis.
    """
    values = np.asarray(y, dtype=float)
    if values.ndim != 1:
        raise ValueError("SNIP works on one spectrum at a time")
    if values.size < 5:
        raise ValueError("hacen falta al menos cinco puntos")
    if not np.all(np.isfinite(values)):
        raise ValueError("hay valores no finitos en el espectro")

    span = int(iterations if iterations is not None else max(2, values.size // 20))
    span = max(1, min(span, (values.size - 1) // 2))

    offset = min(0.0, float(values.min()))
    # Odd-mirrored padding, not clamping. At the first channel of a
    # decaying background, averaging a point with its own value and its
    # neighbour gives something below the curve, so the clipping eats the
    # edge — a third of the background at the first point, with the
    # interior exact. An even reflection halves that and an odd one, which
    # continues the slope instead of turning it around, removes it.
    working = np.pad(_lls(values - offset), span, mode="reflect",
                     reflect_type="odd")

    windows = range(span, 0, -1) if decreasing else range(1, span + 1)
    for width in windows:
        average = 0.5 * (np.roll(working, width) + np.roll(working, -width))
        if smooth_orders:
            kernel = np.ones(2 * smooth_orders + 1) / (2 * smooth_orders + 1)
            average = np.convolve(average, kernel, mode="same")
        average[:width] = working[:width]
        average[-width:] = working[-width:]
        working = np.minimum(working, average)

    return _inverse_lls(working[span:span + values.size]) + offset


def _lls(values: np.ndarray) -> np.ndarray:
    """Log-log-square-root transform. Makes Poisson noise near-constant."""
    return np.log(np.log(np.sqrt(np.clip(values, 0.0, None) + 1.0) + 1.0) + 1.0)


def _inverse_lls(values: np.ndarray) -> np.ndarray:
    return (np.exp(np.exp(values) - 1.0) - 1.0) ** 2 - 1.0


def estimate_baseline(
    spectrum: Spectrum,
    method: str = "asls",
    **kwargs,
) -> np.ndarray:
    """Dispatch to one of the baseline estimators.

    Parameters
    ----------
    spectrum:
        The spectrum to estimate a background for.
    method:
        ``"arpls"``, ``"asls"``, ``"polynomial"`` or ``"rubberband"``.
    **kwargs:
        Passed through to the chosen estimator.

    Returns
    -------
    numpy.ndarray
        The baseline on the spectrum's own axis.
    """
    key = method.lower().replace("-", "").replace("_", "")
    if key in {"arpls", "arls"}:
        return arpls_baseline(spectrum.intensity, **kwargs)
    if key in {"asls", "als"}:
        return asls_baseline(spectrum.intensity, **kwargs)
    if key in {"polynomial", "poly"}:
        return polynomial_baseline(spectrum.shift, spectrum.intensity, **kwargs)
    if key in {"rubberband", "hull", "convexhull"}:
        return rubberband_baseline(spectrum.shift, spectrum.intensity)
    if key == "snip":
        return snip_baseline(spectrum.intensity, **kwargs)
    raise ValueError(
        f"unknown baseline method {method!r}; expected arpls, asls, snip, "
        "polynomial or rubberband"
    )


def subtract_baseline(
    spectrum: Spectrum,
    method: str = "asls",
    clip_negative: bool = False,
    **kwargs,
) -> tuple[Spectrum, np.ndarray]:
    """Remove a background and return both the result and the baseline.

    Parameters
    ----------
    spectrum:
        Input spectrum.
    method:
        See :func:`estimate_baseline`.
    clip_negative:
        Whether to floor the corrected spectrum at zero. Off by default:
        the negative excursions left behind are the honest signature of an
        over-subtracted baseline, and clipping them hides that while also
        biasing every subsequent area upwards.
    **kwargs:
        Estimator parameters.

    Returns
    -------
    (Spectrum, numpy.ndarray)
        The corrected spectrum and the baseline that was subtracted.
    """
    base = estimate_baseline(spectrum, method=method, **kwargs)
    corrected = spectrum.intensity - base
    if clip_negative:
        corrected = np.clip(corrected, 0.0, None)
    args = ", ".join(f"{k}={v!r}" for k, v in sorted(kwargs.items()))
    label = f"baseline({method}{', ' + args if args else ''})"
    return spectrum.with_intensity(corrected, label), base


__all__ = [
    "CUTOFF_BAND_WIDTHS",
    "LAMBDA_LIMITS",
    "MIN_WIDEST_BAND_CM",
    "arpls_baseline",
    "asls_baseline",
    "auto_lambda",
    "background_mask",
    "cutoff_for_lambda",
    "lambda_for_cutoff",
    "estimate_baseline",
    "polynomial_baseline",
    "rubberband_baseline",
    "snip_baseline",
    "subtract_baseline",
]
