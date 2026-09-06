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
    so it slides under the peaks. Two knobs and no peak list needed; this
    is the default and the right first choice for fluorescence.
``polynomial``
    Iterative polynomial fit with peak rejection (Lieber & Mahadevan-Jansen
    2003). Cheap and stable, but a low order cannot follow a curved
    fluorescence tail and a high order starts eating the D band.
``rubberband``
    The lower convex hull. Assumption-free in shape but it only touches the
    data at a few points, so it under-subtracts wherever the true baseline
    is concave — which in the 1000–1800 cm⁻¹ window is most of it.

All three return the *baseline*, not the corrected spectrum; the caller
subtracts. Keeping them separate lets the GUI draw the proposed baseline
over the raw data before anything is committed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .spectrum import Spectrum


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

    # Second-difference operator as a sparse matrix; D.T @ D is the penalty.
    diags = np.array([1.0, -2.0, 1.0])
    d = sparse.diags(
        [diags[0] * np.ones(n - 2), diags[1] * np.ones(n - 2), diags[2] * np.ones(n - 2)],
        offsets=[0, 1, 2],
        shape=(n - 2, n),
        format="csc",
    )
    penalty = lam * (d.T @ d)

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
        ``"asls"``, ``"polynomial"`` or ``"rubberband"``.
    **kwargs:
        Passed through to the chosen estimator.

    Returns
    -------
    numpy.ndarray
        The baseline on the spectrum's own axis.
    """
    key = method.lower().replace("-", "").replace("_", "")
    if key in {"asls", "als"}:
        return asls_baseline(spectrum.intensity, **kwargs)
    if key in {"polynomial", "poly"}:
        return polynomial_baseline(spectrum.shift, spectrum.intensity, **kwargs)
    if key in {"rubberband", "hull", "convexhull"}:
        return rubberband_baseline(spectrum.shift, spectrum.intensity)
    raise ValueError(
        f"unknown baseline method {method!r}; expected asls, polynomial or rubberband"
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
    "asls_baseline",
    "estimate_baseline",
    "polynomial_baseline",
    "rubberband_baseline",
    "subtract_baseline",
]
