"""Non-linear least-squares fitting of a sum of peaks plus a background.

The design decisions worth defending:

**Bounded, not free.** Every parameter is fitted inside physical bounds. An
unbounded fit of five overlapping bands in the 1000–1800 cm⁻¹ window will
happily put the D band at 1480 cm⁻¹ with a width of 400 cm⁻¹, reach a lower
χ² than the physical solution, and report it without complaint. Bounds are
how the physics enters a fit that the data alone cannot constrain.

**A background inside the fit.** Even after global baseline removal, the
1000–1800 cm⁻¹ window sits on a residual slope. Fitting a linear (or
quadratic) term jointly with the peaks stops that slope being absorbed into
the width of the broadest component, which is the single most common way a
published D3 area goes wrong.

**Uncertainties, and their caveats.** Standard errors come from the
Gauss-Newton approximation to the covariance, scaled by the residual
variance. They are only meaningful if the residuals are independent — so
this module *checks the spectrum's history* and warns when the data were
smoothed, which correlates neighbouring residuals and makes every error bar
optimistic, typically by a factor of two or three.

**Degeneracy, reported.** With five overlapping components some parameter
pairs become nearly collinear. The correlation matrix is computed and any
pair above 0.95 is named in ``warnings``. A fit like that is not wrong, but
its individual areas are not independently determined and should not be
quoted to three digits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from ..core.spectrum import Spectrum
from .lineshapes import PROFILES, bwf_peak_height, bwf_peak_position, resolve_profile

#: Background polynomial orders accepted by :class:`FitModel`.
BACKGROUND_ORDERS = {"none": -1, "constant": 0, "linear": 1, "quadratic": 2}


@dataclass
class PeakSpec:
    """One component of a fit: a starting guess plus the bounds around it.

    Parameters
    ----------
    name:
        Label used in the report, e.g. ``"D"``, ``"G"``, ``"D3"``.
    profile:
        A key of :data:`~ramancarbon.models.lineshapes.PROFILES`.
    centre, height, fwhm:
        Initial values. ``height`` is the amplitude parameter (see the note
        on BWF in :mod:`~ramancarbon.models.lineshapes`).
    centre_bounds, height_bounds, fwhm_bounds:
        ``(low, high)`` for each. ``None`` means the fitter picks a
        default: ±25 cm⁻¹ around the guess for the centre, [0, 10×guess]
        for the height, and [2, 400] cm⁻¹ for the width.
    extra:
        Initial values of the profile's extra parameters (η for
        pseudo-Voigt, 1/q for BWF). ``None`` uses the profile default.
    extra_bounds:
        Bounds for those, one ``(low, high)`` per extra parameter.
    fixed:
        Names of parameters to hold constant: any of ``"centre"``,
        ``"height"``, ``"fwhm"`` or an extra parameter's name. Fixing a
        centre to a literature value is a legitimate way to fit a shoulder
        that the data cannot locate on its own — as long as the report says
        it was fixed, which it does.
    band:
        Optional key into the band database, linking this component to a
        named physical mode. Set automatically by the deconvolution presets.
    """

    name: str
    profile: str = "lorentzian"
    centre: float = 0.0
    height: float = 1.0
    fwhm: float = 30.0
    centre_bounds: Optional[tuple[float, float]] = None
    height_bounds: Optional[tuple[float, float]] = None
    fwhm_bounds: Optional[tuple[float, float]] = None
    extra: Optional[tuple[float, ...]] = None
    extra_bounds: Optional[tuple[tuple[float, float], ...]] = None
    fixed: tuple[str, ...] = ()
    band: Optional[str] = None

    def __post_init__(self) -> None:
        self.profile = resolve_profile(self.profile)
        spec = PROFILES[self.profile]
        if self.extra is None:
            self.extra = tuple(spec["defaults"])
        elif len(self.extra) != len(spec["defaults"]):
            raise ValueError(
                f"profile {self.profile!r} takes {len(spec['defaults'])} extra "
                f"parameter(s), got {len(self.extra)}"
            )
        if self.extra_bounds is None:
            self.extra_bounds = tuple(spec["bounds"])
        if self.fwhm <= 0:
            raise ValueError(f"peak {self.name!r}: fwhm must be positive")

    @property
    def extra_names(self) -> tuple[str, ...]:
        """Names of this profile's extra parameters."""
        return tuple(PROFILES[self.profile]["extra"])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """The component's own curve at its current parameter values."""
        fn = PROFILES[self.profile]["function"]
        return fn(x, self.centre, self.height, self.fwhm, *self.extra)


@dataclass
class FittedPeak:
    """One component after fitting, with derived quantities and errors."""

    name: str
    profile: str
    band: Optional[str]
    centre: float
    height: float
    fwhm: float
    extra: tuple[float, ...]
    extra_names: tuple[str, ...]
    area: float
    peak_position: float
    """Position of the curve's maximum. Differs from ``centre`` only for BWF."""
    peak_height: float
    """Value of the curve at its maximum. Differs from ``height`` only for BWF."""
    errors: dict[str, float] = field(default_factory=dict)
    fixed: tuple[str, ...] = ()

    def curve(self, x: np.ndarray) -> np.ndarray:
        """Evaluate this fitted component on an arbitrary grid."""
        fn = PROFILES[self.profile]["function"]
        return fn(x, self.centre, self.height, self.fwhm, *self.extra)

    def summary(self) -> str:
        """One line for the report table."""
        err = self.errors.get("centre")
        pos = f"{self.peak_position:.1f}" + (f" ± {err:.1f}" if err else "")
        extras = ", ".join(
            f"{n}={v:.3f}" for n, v in zip(self.extra_names, self.extra)
        )
        tail = f", {extras}" if extras else ""
        return (
            f"{self.name:>4s}  {pos:>14s} cm⁻¹  I={self.peak_height:.4g}  "
            f"FWHM={self.fwhm:.1f}  A={self.area:.4g}{tail}"
        )


@dataclass
class FitResult:
    """Everything a deconvolution produced, including why to distrust it."""

    peaks: list[FittedPeak]
    background: np.ndarray
    background_coefficients: np.ndarray
    x: np.ndarray
    y: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    r_squared: float
    reduced_chi2: float
    aic: float
    bic: float
    n_parameters: int
    success: bool
    message: str
    warnings: list[str] = field(default_factory=list)
    correlations: dict[tuple[str, str], float] = field(default_factory=dict)

    def peak(self, name: str) -> Optional[FittedPeak]:
        """Look a component up by name, or by band key."""
        for p in self.peaks:
            if p.name == name or p.band == name:
                return p
        return None

    def total_area(self) -> float:
        """Sum of the component areas (background excluded)."""
        return float(sum(p.area for p in self.peaks))

    def summary(self) -> str:
        """Multi-line human-readable report of the fit."""
        lines = [
            f"Ajuste: {len(self.peaks)} componentes, {self.n_parameters} parámetros",
            f"R² = {self.r_squared:.5f}   χ²_red = {self.reduced_chi2:.4g}   "
            f"AIC = {self.aic:.1f}   BIC = {self.bic:.1f}",
            "",
        ]
        lines.extend(p.summary() for p in self.peaks)
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


@dataclass
class FitModel:
    """A set of peaks plus a polynomial background, ready to be fitted."""

    peaks: list[PeakSpec]
    window: tuple[float, float]
    background: str = "linear"
    name: str = "modelo"

    def __post_init__(self) -> None:
        if not self.peaks:
            raise ValueError("a fit model needs at least one peak")
        if self.background not in BACKGROUND_ORDERS:
            raise ValueError(
                f"unknown background {self.background!r}; "
                f"expected one of {', '.join(BACKGROUND_ORDERS)}"
            )
        lo, hi = self.window
        if hi <= lo:
            raise ValueError(f"empty fit window [{lo}, {hi}]")
        outside = [p.name for p in self.peaks if not lo - 50 <= p.centre <= hi + 50]
        if outside:
            raise ValueError(
                "these components start outside the fit window "
                f"[{lo:g}, {hi:g}] cm⁻¹: {', '.join(outside)}"
            )

    @property
    def background_order(self) -> int:
        """Polynomial degree, or -1 for no background."""
        return BACKGROUND_ORDERS[self.background]


def _pack(model: FitModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, str]]]:
    """Flatten the free parameters into a vector with bounds.

    Returns ``(x0, lower, upper, layout)`` where ``layout`` maps each free
    slot to ``(peak index, parameter name)``. Fixed parameters simply do not
    appear, which is cleaner than fitting them with equal bounds — scipy's
    trust-region solver handles a zero-width interval poorly.
    """
    x0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    layout: list[tuple[int, str]] = []

    for i, spec in enumerate(model.peaks):
        defaults = {
            "centre": (spec.centre, spec.centre_bounds or (spec.centre - 25.0, spec.centre + 25.0)),
            "height": (spec.height, spec.height_bounds or (0.0, max(abs(spec.height) * 10.0, 1e-6))),
            "fwhm": (spec.fwhm, spec.fwhm_bounds or (2.0, 400.0)),
        }
        for key in ("centre", "height", "fwhm"):
            if key in spec.fixed:
                continue
            value, (lo, hi) = defaults[key]
            x0.append(float(np.clip(value, lo, hi)))
            lower.append(float(lo))
            upper.append(float(hi))
            layout.append((i, key))
        for j, ename in enumerate(spec.extra_names):
            if ename in spec.fixed:
                continue
            lo, hi = spec.extra_bounds[j]
            x0.append(float(np.clip(spec.extra[j], lo, hi)))
            lower.append(float(lo))
            upper.append(float(hi))
            layout.append((i, ename))

    order = model.background_order
    for k in range(order + 1):
        x0.append(0.0)
        lower.append(-np.inf)
        upper.append(np.inf)
        layout.append((-1, f"bg{k}"))

    return np.asarray(x0), np.asarray(lower), np.asarray(upper), layout


def _unpack(
    params: np.ndarray, model: FitModel, layout: list[tuple[int, str]]
) -> tuple[list[dict], np.ndarray]:
    """Turn a parameter vector back into per-peak dictionaries and bg coefficients."""
    values: list[dict] = []
    for spec in model.peaks:
        entry = {
            "centre": spec.centre,
            "height": spec.height,
            "fwhm": spec.fwhm,
        }
        for name, value in zip(spec.extra_names, spec.extra):
            entry[name] = value
        values.append(entry)
    background: list[float] = []
    for value, (index, key) in zip(params, layout):
        if index < 0:
            background.append(float(value))
        else:
            values[index][key] = float(value)
    return values, np.asarray(background)


def _evaluate(
    params: np.ndarray,
    x: np.ndarray,
    model: FitModel,
    layout: list[tuple[int, str]],
    x_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Model curve and its background at the given parameters."""
    values, bg_coeffs = _unpack(params, model, layout)
    total = np.zeros_like(x)
    for spec, entry in zip(model.peaks, values):
        fn = PROFILES[spec.profile]["function"]
        extras = [entry[n] for n in spec.extra_names]
        total = total + fn(x, entry["centre"], entry["height"], entry["fwhm"], *extras)
    if bg_coeffs.size:
        background = np.polyval(bg_coeffs[::-1], x_scaled)
    else:
        background = np.zeros_like(x)
    return total + background, background


def fit_model(
    spectrum: Spectrum,
    model: FitModel,
    max_nfev: int = 6000,
    loss: str = "linear",
    f_scale: float = 1.0,
) -> FitResult:
    """Fit a :class:`FitModel` to a spectrum over the model's window.

    Parameters
    ----------
    spectrum:
        The spectrum to fit. Baseline-correct it first; the in-fit
        background is meant to absorb a residual slope, not a fluorescence
        hump.
    model:
        Components, bounds and window.
    max_nfev:
        Cap on function evaluations. A five-component fit converges in a
        few hundred; hitting the cap means the model is under-constrained.
    loss:
        scipy's robust loss. ``"linear"`` is ordinary least squares;
        ``"soft_l1"`` downweights outliers and is useful when a cosmic ray
        survived despiking.
    f_scale:
        Soft-threshold for the robust losses, in intensity units.

    Returns
    -------
    FitResult

    Raises
    ------
    ValueError
        If the window contains fewer points than there are free parameters.
    """
    lo, hi = model.window
    x, y = spectrum.region(lo, hi)
    if x.size < 5:
        raise ValueError(
            f"the fit window [{lo:g}, {hi:g}] cm⁻¹ contains only {x.size} points; "
            f"the spectrum spans {spectrum.range[0]:.0f}–{spectrum.range[1]:.0f} cm⁻¹"
        )

    x0, lower, upper, layout = _pack(model)
    if x.size <= x0.size:
        raise ValueError(
            f"{x0.size} free parameters but only {x.size} data points in "
            f"[{lo:g}, {hi:g}] cm⁻¹ — the fit is underdetermined"
        )

    # Centre and scale the abscissa for the background polynomial only; the
    # peak centres stay in cm^-1 so their bounds remain readable.
    x_mid = 0.5 * (x[0] + x[-1])
    x_half = max(0.5 * (x[-1] - x[0]), 1e-9)
    x_scaled = (x - x_mid) / x_half

    def residual(params: np.ndarray) -> np.ndarray:
        curve, _ = _evaluate(params, x, model, layout, x_scaled)
        return curve - y

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        loss=loss,
        f_scale=f_scale,
        x_scale="jac",
    )

    fitted, background = _evaluate(result.x, x, model, layout, x_scaled)
    resid = y - fitted
    n, k = x.size, result.x.size
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = max(n - k, 1)
    reduced_chi2 = ss_res / dof
    # Gaussian log-likelihood form of the information criteria.
    aic = n * np.log(max(ss_res / n, 1e-300)) + 2 * k
    bic = n * np.log(max(ss_res / n, 1e-300)) + k * np.log(n)

    errors, correlations = _uncertainties(result, resid, layout, model)
    values, bg_coeffs = _unpack(result.x, model, layout)

    peaks: list[FittedPeak] = []
    for i, (spec, entry) in enumerate(zip(model.peaks, values)):
        extras = tuple(entry[nm] for nm in spec.extra_names)
        area = float(PROFILES[spec.profile]["area"](entry["height"], entry["fwhm"], extras))
        if spec.profile == "bwf":
            position = bwf_peak_position(entry["centre"], entry["fwhm"], extras[0])
            apex = bwf_peak_height(entry["height"], extras[0])
        else:
            position, apex = entry["centre"], entry["height"]
        peaks.append(
            FittedPeak(
                name=spec.name,
                profile=spec.profile,
                band=spec.band,
                centre=entry["centre"],
                height=entry["height"],
                fwhm=entry["fwhm"],
                extra=extras,
                extra_names=spec.extra_names,
                area=area,
                peak_position=position,
                peak_height=apex,
                errors=errors.get(i, {}),
                fixed=spec.fixed,
            )
        )

    warnings = _collect_warnings(spectrum, model, result, peaks, correlations, max_nfev)

    return FitResult(
        peaks=peaks,
        background=background,
        background_coefficients=bg_coeffs,
        x=x,
        y=y,
        fitted=fitted,
        residual=resid,
        r_squared=r_squared,
        reduced_chi2=reduced_chi2,
        aic=float(aic),
        bic=float(bic),
        n_parameters=k,
        success=bool(result.success),
        message=str(result.message),
        warnings=warnings,
        correlations=correlations,
    )


def _uncertainties(
    result, resid: np.ndarray, layout: list[tuple[int, str]], model: FitModel
) -> tuple[dict[int, dict[str, float]], dict[tuple[str, str], float]]:
    """Standard errors and the parameter correlation matrix.

    ``cov = s² (JᵀJ)⁻¹`` with ``s² = SSR / (n − k)``. A singular JᵀJ means
    the model is degenerate at the solution; rather than returning a
    pseudo-inverse and pretending, the errors are simply omitted and the
    caller warns.
    """
    errors: dict[int, dict[str, float]] = {}
    correlations: dict[tuple[str, str], float] = {}
    jac = getattr(result, "jac", None)
    if jac is None or jac.size == 0:
        return errors, correlations
    n, k = jac.shape
    if n <= k:
        return errors, correlations
    try:
        jtj = jac.T @ jac
        cov = np.linalg.inv(jtj) * (float(np.sum(resid**2)) / (n - k))
    except np.linalg.LinAlgError:
        return errors, correlations

    diag = np.diag(cov)
    if np.any(diag < 0) or not np.all(np.isfinite(diag)):
        return errors, correlations
    sigma = np.sqrt(diag)
    for value, (index, key) in zip(sigma, layout):
        if index < 0:
            continue
        errors.setdefault(index, {})[key] = float(value)

    labels = [
        f"{model.peaks[i].name}.{key}" if i >= 0 else key for i, key in layout
    ]
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.outer(sigma, sigma)
        corr = np.where(denom > 0, cov / denom, 0.0)
    for a in range(k):
        for b in range(a + 1, k):
            value = float(corr[a, b])
            if np.isfinite(value) and abs(value) > 0.95:
                correlations[(labels[a], labels[b])] = value
    return errors, correlations


def _collect_warnings(
    spectrum: Spectrum,
    model: FitModel,
    result,
    peaks: list[FittedPeak],
    correlations: dict[tuple[str, str], float],
    max_nfev: int,
) -> list[str]:
    """Everything the user should know before quoting these numbers."""
    warnings: list[str] = []
    if not result.success:
        warnings.append(f"el ajuste no convergió: {result.message}")
    if getattr(result, "nfev", 0) >= max_nfev:
        warnings.append(
            "se agotó el número de evaluaciones; el modelo probablemente tiene "
            "más componentes de los que los datos pueden determinar"
        )
    if any("smooth" in step for step in spectrum.history):
        warnings.append(
            "el espectro fue suavizado antes del ajuste: los residuos están "
            "correlacionados y las incertidumbres mostradas son optimistas "
            "(típicamente por un factor 2–3)"
        )
    for spec, fitted in zip(model.peaks, peaks):
        for key, bounds in (
            ("centre", spec.centre_bounds or (spec.centre - 25.0, spec.centre + 25.0)),
            ("fwhm", spec.fwhm_bounds or (2.0, 400.0)),
        ):
            value = getattr(fitted, key)
            span = bounds[1] - bounds[0]
            if span > 0 and min(value - bounds[0], bounds[1] - value) < 0.01 * span:
                warnings.append(
                    f"{spec.name}: el parámetro «{key}» quedó pegado a su límite "
                    f"({value:.1f}); ensancha los límites o revisa el modelo"
                )
    if correlations:
        pairs = ", ".join(f"{a}–{b} ({v:+.2f})" for (a, b), v in list(correlations.items())[:4])
        warnings.append(
            "parámetros casi degenerados: " + pairs + ". Las áreas individuales "
            "no están determinadas de forma independiente"
        )
    if not any(p.errors for p in peaks):
        warnings.append(
            "no se pudieron estimar incertidumbres (matriz normal singular): "
            "el modelo es degenerado en la solución"
        )
    return warnings


__all__ = [
    "BACKGROUND_ORDERS",
    "FitModel",
    "FitResult",
    "FittedPeak",
    "PeakSpec",
    "fit_model",
]
