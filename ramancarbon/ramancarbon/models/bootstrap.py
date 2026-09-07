"""Realistic uncertainties, by refitting resampled data.

The standard errors a least-squares fit reports come from the curvature of
χ² at the minimum. They are the right answer to a narrow question — how
sharply does χ² rise if I move this one parameter, with the others free —
and they are systematically too small for the number people actually want,
which is "how much would I_D/I_G change if I measured this sample again".

They miss three things:

* the correlation structure between parameters, when what you want is a
  *derived* quantity like a ratio of two areas from different components;
* any correlation in the residuals, which smoothing puts there by
  construction and an imperfect model puts there by omission;
* the fact that the fit itself might land somewhere slightly different.

Resampling gets all three for free. The residual is treated as an estimate
of the noise, resampled, added back onto the fitted curve, and the whole fit
re-run. The spread of the results over many such synthetic datasets is a
direct estimate of how reproducible the fit is.

This costs one fit per replicate, so it is opt-in, not automatic. Fifty
replicates on a four-component model takes a couple of seconds and is
enough for a confidence interval quoted to two significant figures.

**What it does not cover.** Everything that is not noise: baseline choice,
which components are in the model, whether the sample is homogeneous. Those
dominate in practice. A tight bootstrap interval means the fit is stable,
not that the number is right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..core.spectrum import Spectrum
from .fitting import FitModel, FitResult, fit_model

#: Default number of resampled datasets.
DEFAULT_REPLICATES = 60


@dataclass
class Interval:
    """A bootstrap estimate of one quantity."""

    name: str
    value: float
    """The value from the fit to the real data."""
    median: float
    low: float
    high: float
    """Percentile bounds of the bootstrap distribution."""
    std: float
    n: int
    level: float = 0.95

    @property
    def half_width(self) -> float:
        """Half the interval, for quoting as ``value ± half_width``."""
        return 0.5 * (self.high - self.low)

    @property
    def relative(self) -> float:
        """Half-width as a fraction of the value."""
        return self.half_width / abs(self.value) if self.value else float("nan")

    def __str__(self) -> str:
        return (
            f"{self.name} = {self.value:.4g}  "
            f"[{self.low:.4g}, {self.high:.4g}] al {self.level * 100:.0f} % "
            f"(±{self.half_width:.3g}, {self.relative * 100:.1f} %)"
        )


@dataclass
class BootstrapResult:
    """Intervals for every quantity that was tracked."""

    intervals: dict[str, Interval] = field(default_factory=dict)
    replicates: int = 0
    failures: int = 0
    warnings: list[str] = field(default_factory=list)

    def get(self, name: str) -> Optional[Interval]:
        return self.intervals.get(name)

    def summary(self) -> str:
        lines = [
            f"Remuestreo: {self.replicates} réplicas"
            + (f", {self.failures} sin converger" if self.failures else "")
        ]
        lines.extend("  " + str(i) for i in self.intervals.values())
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


def bootstrap_fit(
    spectrum: Spectrum,
    model: FitModel,
    quantities: dict[str, Callable[[FitResult], Optional[float]]],
    replicates: int = DEFAULT_REPLICATES,
    level: float = 0.95,
    seed: int = 0,
    block: Optional[int] = None,
) -> BootstrapResult:
    """Estimate uncertainties by refitting resampled residuals.

    Parameters
    ----------
    spectrum:
        The spectrum that was fitted.
    model:
        The model to refit on each replicate. Its starting values are
        reused every time, so each replicate is an independent fit rather
        than a walk away from the previous one.
    quantities:
        ``{name: function(FitResult) -> float or None}``. Anything
        computable from a fit can be tracked — a ratio of areas, a band
        position, a width. Replicates where a function returns ``None`` are
        skipped for that quantity only.
    replicates:
        How many synthetic datasets. Below ~30 the percentile bounds are
        themselves noisy.
    level:
        Confidence level for the interval.
    seed:
        Seed for the resampling, so a reported interval is reproducible.
    block:
        Length of the blocks used when resampling residuals. Point-wise
        resampling assumes the residuals are independent; if the spectrum
        was smoothed, or the model is missing a component, they are not,
        and independent resampling then destroys exactly the correlation
        that makes the real uncertainty larger. ``None`` picks a block
        length from the residual's own autocorrelation.

    Returns
    -------
    BootstrapResult

    Raises
    ------
    ValueError
        If the model cannot be fitted to the original data at all.
    """
    base = fit_model(spectrum, model)
    residual = base.residual
    n = residual.size
    if n < 20:
        raise ValueError("hacen falta al menos 20 puntos para remuestrear")

    length = block if block is not None else _block_length(residual)
    rng = np.random.default_rng(seed)

    observed = {name: fn(base) for name, fn in quantities.items()}
    samples: dict[str, list[float]] = {name: [] for name in quantities}
    failures = 0

    for _ in range(int(replicates)):
        synthetic = base.fitted + _resample(residual, length, rng)
        replicate = spectrum.copy()
        # Write the synthetic data back onto the fit window only; the rest
        # of the spectrum plays no part in this model.
        mask = (spectrum.shift >= base.x[0]) & (spectrum.shift <= base.x[-1])
        replicate.intensity = spectrum.intensity.copy()
        replicate.intensity[mask] = synthetic
        try:
            fitted = fit_model(replicate, model)
        except (ValueError, np.linalg.LinAlgError):
            failures += 1
            continue
        for name, function in quantities.items():
            value = function(fitted)
            if value is not None and np.isfinite(value):
                samples[name].append(float(value))

    result = BootstrapResult(replicates=int(replicates), failures=failures)
    tail = 0.5 * (1.0 - level)
    for name, values in samples.items():
        if len(values) < max(10, replicates // 4):
            continue
        array = np.asarray(values, dtype=float)
        centre = observed.get(name)
        result.intervals[name] = Interval(
            name=name,
            value=float(centre) if centre is not None else float(np.median(array)),
            median=float(np.median(array)),
            low=float(np.quantile(array, tail)),
            high=float(np.quantile(array, 1.0 - tail)),
            std=float(np.std(array, ddof=1)),
            n=int(array.size),
            level=level,
        )

    if length > 1:
        result.warnings.append(
            f"los residuos están correlacionados a lo largo de {length} puntos, "
            "así que se remuestrean en bloques de esa longitud. Remuestrear "
            "punto a punto habría destruido esa correlación y devuelto "
            "intervalos demasiado estrechos"
        )
    result.warnings.append(
        "esto mide la reproducibilidad del AJUSTE frente al ruido. No incluye "
        "la elección de línea base, ni qué componentes lleva el modelo, ni la "
        "heterogeneidad de la muestra — que en la práctica dominan"
    )
    if failures:
        result.warnings.append(
            f"{failures} réplicas no convergieron; si son muchas, el modelo es "
            "inestable y el intervalo se queda corto"
        )
    return result


def _block_length(residual: np.ndarray, max_block: int = 64) -> int:
    """Block length from where the residual's autocorrelation first crosses 1/e.

    A residual that is white gives 1 and the resampling is point-wise. A
    smoothed or under-modelled residual gives more, and the blocks preserve
    that structure through the resampling.
    """
    r = residual - residual.mean()
    denominator = float(np.sum(r * r))
    if denominator <= 0:
        return 1
    limit = min(max_block, r.size // 4)
    for lag in range(1, max(limit, 2)):
        rho = float(np.sum(r[:-lag] * r[lag:]) / denominator)
        if rho < np.exp(-1.0):
            return max(1, lag)
    return max(1, limit)


def _resample(residual: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a residual sequence of the same length, in blocks."""
    n = residual.size
    if block <= 1:
        return rng.choice(residual, size=n, replace=True)
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([residual[s : s + block] for s in starts])[:n]


def ratio_of(numerator: str, denominator: str, use_area: bool = True):
    """A quantity function for :func:`bootstrap_fit`: one band over another.

    Parameters
    ----------
    numerator, denominator:
        Component names, as they appear in the model.
    use_area:
        Ratio of integrated areas rather than of peak heights.

    Returns
    -------
    callable
        Suitable as a value of the ``quantities`` mapping.
    """

    def extract(fit: FitResult) -> Optional[float]:
        top, bottom = fit.peak(numerator), fit.peak(denominator)
        if top is None or bottom is None:
            return None
        a = top.area if use_area else top.peak_height
        b = bottom.area if use_area else bottom.peak_height
        if not b:
            return None
        return float(a / b)

    return extract


def position_of(name: str):
    """A quantity function: the position of one component."""

    def extract(fit: FitResult) -> Optional[float]:
        peak = fit.peak(name)
        return float(peak.peak_position) if peak else None

    return extract


def width_of(name: str):
    """A quantity function: the FWHM of one component."""

    def extract(fit: FitResult) -> Optional[float]:
        peak = fit.peak(name)
        return float(peak.fwhm) if peak else None

    return extract


__all__ = [
    "DEFAULT_REPLICATES",
    "BootstrapResult",
    "Interval",
    "bootstrap_fit",
    "position_of",
    "ratio_of",
    "width_of",
]
