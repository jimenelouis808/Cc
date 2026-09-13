"""Turning a cube into a picture: one number per pixel.

Everything a Raman map is used for is a per-pixel number — the intensity
of a band, a ratio of two, a band's position, its width — drawn as an
image. Each of them has a way of producing a beautiful picture of nothing,
and each of those is guarded here.

**An area needs its own local baseline.** Integrating raw counts between
two limits integrates the fluorescence under them too, and fluorescence
varies across a sample far more than the Raman signal does. A "map of the
D band" made from raw areas is usually a map of the background slope.
A straight line between the window's own ends is subtracted first.

**A position must be interpolated, not picked.** ``argmax`` can only
return values on the spectral grid, so with a 2 cm⁻¹ step a shift map
comes out in terraces, and terraces read as domains. A parabola through
the three points around the maximum gives a continuous position.

**A ratio has to be masked where its denominator is noise.** Dividing by
a band that is not there produces the brightest pixels in the image, and
they are all in the empty part of the sample.

**A width means nothing on overlapping bands.** The full width at half
maximum measured by crossings is reported with the warning that where D
and G overlap — which is everywhere in carbon — it is the width of the
pair.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.compat import trapezoid
from .cube import MapError, RamanMap


@dataclass
class PropertyMap:
    """One number per pixel, with what it is and what it cannot say."""

    values: np.ndarray
    """``(n_y, n_x)``. NaN where the pixel has no data or was masked."""
    label: str
    units: str = ""
    cube: Optional[RamanMap] = None
    warnings: list[str] = field(default_factory=list)
    masked: int = 0
    """How many pixels were dropped for failing the test that protects
    this quantity. A number worth showing: half the map masked means the
    band is not there."""

    @property
    def extent(self) -> Optional[tuple[float, float, float, float]]:
        return self.cube.extent if self.cube else None

    @property
    def finite(self) -> np.ndarray:
        return self.values[np.isfinite(self.values)]

    def statistics(self) -> dict[str, float]:
        """Mean, median, spread and range over the pixels that survived."""
        good = self.finite
        if good.size == 0:
            return {}
        return {
            "media": float(np.mean(good)),
            "mediana": float(np.median(good)),
            "desviacion": float(np.std(good, ddof=1)) if good.size > 1 else 0.0,
            "minimo": float(np.min(good)),
            "maximo": float(np.max(good)),
            "pixeles": int(good.size),
        }

    def describe(self) -> str:
        stats = self.statistics()
        if not stats:
            return f"{self.label}: ningún píxel válido"
        text = (f"{self.label} = {stats['mediana']:.4g} ± {stats['desviacion']:.2g} "
                f"{self.units} (mediana ± desviación, {stats['pixeles']} píxeles)")
        if self.masked:
            text += f", {self.masked} enmascarados"
        return text


@contextmanager
def _quiet_missing():
    """Silence NumPy's all-NaN warning.

    A missing pixel is an expected state of a map, not a surprise: it is
    stored as NaN on purpose and every result here is re-masked afterwards.
    Letting NumPy warn once per reduction buries the warnings that matter.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        yield


def _window(cube: RamanMap, low: float, high: float) -> np.ndarray:
    keep = (cube.shift >= low) & (cube.shift <= high)
    if np.count_nonzero(keep) < 3:
        raise MapError(
            f"la ventana {low}–{high} cm⁻¹ solo tiene "
            f"{int(np.count_nonzero(keep))} puntos del espectro"
        )
    return keep


def _local_baseline(x: np.ndarray, block: np.ndarray) -> np.ndarray:
    """A straight line through the window's own ends, per pixel.

    Two-point, not fitted: the ends are where the band is not, and fitting
    a line through the whole window fits the band.
    """
    ends = np.stack([block[..., :3].mean(axis=-1), block[..., -3:].mean(axis=-1)],
                    axis=-1)
    fraction = (x - x[0]) / (x[-1] - x[0])
    return ends[..., :1] + (ends[..., 1:] - ends[..., :1]) * fraction


def band_intensity(
    cube: RamanMap,
    low: float,
    high: float,
    mode: str = "area",
    baseline: bool = True,
    label: str = "",
) -> PropertyMap:
    """How much signal there is in a spectral window, per pixel.

    Parameters
    ----------
    mode:
        ``"area"`` integrates; ``"altura"`` takes the maximum. Area is the
        default because it is what survives a change of width — two
        samples with the same amount of material and different crystallite
        sizes have the same area and different heights.
    baseline:
        Subtract a straight line through the window's ends first. Leave it
        on unless the map is already baseline-corrected.
    """
    keep = _window(cube, low, high)
    x = cube.shift[keep]
    block = cube.intensity[:, :, keep]
    if baseline:
        block = block - _local_baseline(x, block)
    with np.errstate(invalid="ignore"), _quiet_missing():
        if mode == "area":
            values = trapezoid(block, x, axis=-1)
            units = "cuentas·cm⁻¹"
        elif mode in ("altura", "height", "max"):
            values = np.nanmax(block, axis=-1)
            units = "cuentas"
        else:
            raise MapError(f"modo desconocido: {mode!r}; usa «area» o «altura»")
    values = np.where(cube.missing, np.nan, values)
    warnings = []
    if not baseline:
        warnings.append(
            "sin línea base local: lo que se integra incluye el fondo, y la "
            "fluorescencia varía por la muestra más que la señal Raman"
        )
    return PropertyMap(
        values=values, label=label or f"{mode} {low:.0f}–{high:.0f} cm⁻¹",
        units=units, cube=cube, warnings=warnings,
    )


def band_ratio(
    cube: RamanMap,
    numerator: tuple[float, float],
    denominator: tuple[float, float],
    mode: str = "area",
    min_signal_to_noise: float = 3.0,
    label: str = "",
) -> PropertyMap:
    """The ratio of two bands, masked where the denominator is not there.

    An unmasked ratio map has its brightest pixels where the sample is
    absent, because that is where the denominator is closest to zero.
    """
    top = band_intensity(cube, *numerator, mode=mode)
    bottom = band_intensity(cube, *denominator, mode=mode)
    noise = noise_level(cube)
    scale = (denominator[1] - denominator[0]) if mode == "area" else 1.0
    threshold = min_signal_to_noise * noise.values * max(scale, 1.0) ** 0.5

    with np.errstate(invalid="ignore", divide="ignore"):
        values = top.values / bottom.values
    weak = ~(bottom.values > threshold)
    values = np.where(weak, np.nan, values)
    masked = int(np.count_nonzero(weak & np.isfinite(bottom.values)))

    warnings = list(top.warnings)
    if masked:
        fraction = masked / max(cube.n_pixels, 1)
        warnings.append(
            f"{masked} píxeles ({100 * fraction:.0f} %) tienen el denominador "
            f"por debajo de {min_signal_to_noise:g}σ y se han enmascarado"
        )
        if fraction > 0.5:
            warnings.append(
                "más de la mitad del mapa: la banda del denominador "
                "prácticamente no está, y el cociente no significa nada donde sí"
            )
    return PropertyMap(
        values=values,
        label=label or (f"I({numerator[0]:.0f}–{numerator[1]:.0f}) / "
                        f"I({denominator[0]:.0f}–{denominator[1]:.0f})"),
        units="", cube=cube, warnings=warnings, masked=masked,
    )


def band_position(
    cube: RamanMap,
    low: float,
    high: float,
    baseline: bool = True,
    label: str = "",
) -> PropertyMap:
    """Where a band's maximum is, to better than the spectral step.

    A parabola through the three points around the maximum. Without it the
    map can only take values on the grid, and a shift map in terraces is
    read as a map of domains.
    """
    keep = _window(cube, low, high)
    x = cube.shift[keep]
    block = cube.intensity[:, :, keep]
    if baseline:
        block = block - _local_baseline(x, block)

    with np.errstate(invalid="ignore"), _quiet_missing():
        peak = np.nanargmax(np.where(np.isfinite(block), block, -np.inf), axis=-1)
    peak = np.clip(peak, 1, x.size - 2)
    rows, columns = np.indices(peak.shape)
    left = block[rows, columns, peak - 1]
    centre = block[rows, columns, peak]
    right = block[rows, columns, peak + 1]

    denominator = left - 2 * centre + right
    with np.errstate(invalid="ignore", divide="ignore"):
        offset = 0.5 * (left - right) / denominator
    offset = np.where(np.abs(denominator) < 1e-12, 0.0, offset)
    offset = np.clip(offset, -1.0, 1.0)
    step = np.gradient(x)[peak]
    values = x[peak] + offset * step
    values = np.where(cube.missing, np.nan, values)

    warnings = []
    if x.size and float(np.median(np.diff(x))) > 3.0:
        warnings.append(
            f"el paso espectral es {float(np.median(np.diff(x))):.1f} cm⁻¹; "
            "la interpolación parabólica da una posición continua, pero la "
            "incertidumbre sigue siendo del orden del paso"
        )
    return PropertyMap(
        values=values, label=label or f"posición {low:.0f}–{high:.0f} cm⁻¹",
        units="cm⁻¹", cube=cube, warnings=warnings,
    )


def band_width(
    cube: RamanMap,
    low: float,
    high: float,
    label: str = "",
) -> PropertyMap:
    """Full width at half maximum, by where the band crosses half its top.

    Fast and assumption-free, and only meaningful for an isolated band.
    Where two bands overlap — D and G in any carbon — this measures the
    pair, and says so.
    """
    keep = _window(cube, low, high)
    x = cube.shift[keep]
    block = cube.intensity[:, :, keep]
    block = block - _local_baseline(x, block)

    with np.errstate(invalid="ignore"), _quiet_missing():
        top = np.nanmax(block, axis=-1)
    half = top[..., None] / 2.0
    above = block >= half
    counts = np.count_nonzero(above, axis=-1)
    step = float(np.median(np.diff(x))) if x.size > 1 else np.nan
    values = counts * step
    touching = (above[..., 0] | above[..., -1])
    values = np.where(cube.missing | (counts < 2), np.nan, values)
    values = np.where(touching, np.nan, values)
    masked = int(np.count_nonzero(touching & ~cube.missing))

    warnings = [
        "la anchura a media altura por cruces solo tiene sentido en una "
        "banda aislada; donde D y G se solapan mide el par"
    ]
    if masked:
        warnings.append(
            f"{masked} píxeles tienen la banda tocando el borde de la ventana "
            f"{low:.0f}–{high:.0f} cm⁻¹ y se han enmascarado: la anchura "
            "estaría recortada por la ventana, no por la banda"
        )
    return PropertyMap(
        values=values, label=label or f"FWHM {low:.0f}–{high:.0f} cm⁻¹",
        units="cm⁻¹", cube=cube, warnings=warnings, masked=masked,
    )


def noise_level(cube: RamanMap, window: int = 15) -> PropertyMap:
    """Per-pixel noise, from the median absolute second difference.

    Robust: a second difference kills any smooth background and any linear
    trend, and the median ignores the peaks. The 0.6745 and √6 turn a
    median absolute deviation of second differences into a standard
    deviation.
    """
    block = cube.intensity
    if block.shape[-1] < 5:
        raise MapError("hacen falta al menos cinco puntos espectrales")
    second = np.diff(block, n=2, axis=-1)
    with np.errstate(invalid="ignore"), _quiet_missing():
        mad = np.nanmedian(np.abs(second), axis=-1)
    values = mad / (0.6745 * np.sqrt(6.0))
    values = np.where(cube.missing, np.nan, values)
    return PropertyMap(values=values, label="ruido", units="cuentas", cube=cube)


def signal_to_noise(cube: RamanMap, low: float, high: float) -> PropertyMap:
    """Peak height over noise, per pixel — the map that says where to look."""
    height = band_intensity(cube, low, high, mode="altura")
    noise = noise_level(cube)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = height.values / noise.values
    return PropertyMap(
        values=values, label=f"S/R {low:.0f}–{high:.0f} cm⁻¹",
        units="", cube=cube,
    )


def coverage(cube: RamanMap, low: float, high: float,
             threshold: float = 5.0) -> tuple[PropertyMap, float]:
    """Where the material is at all, and what fraction of the map that is.

    Reported before anything else: a ratio, a cluster map or a histogram
    computed over the empty part of a substrate is a picture of the noise.
    """
    ratio = signal_to_noise(cube, low, high)
    present = ratio.values >= threshold
    fraction = float(np.count_nonzero(present) / max(cube.n_pixels, 1))
    return (
        PropertyMap(
            values=np.where(np.isfinite(ratio.values), present.astype(float), np.nan),
            label=f"cobertura (S/R ≥ {threshold:g})", units="", cube=cube,
            warnings=([] if fraction > 0.2 else [
                f"solo el {100 * fraction:.0f} % del mapa tiene señal por "
                "encima del umbral: lo que se mida en el resto es ruido"]),
        ),
        fraction,
    )


__all__ = [
    "PropertyMap",
    "band_intensity",
    "band_position",
    "band_ratio",
    "band_width",
    "coverage",
    "noise_level",
    "signal_to_noise",
]
