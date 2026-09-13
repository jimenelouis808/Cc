"""A Raman map: thousands of spectra on a grid.

A map is not a list of spectra with coordinates attached. It is a cube —
two spatial axes and one spectral — and treating it as a cube is what
makes the useful operations cheap. Extracting a band's intensity across
40 000 pixels is one array operation on the cube and 40 000 function calls
on a list, and that difference is the difference between a map that
redraws as you move a slider and one that takes a minute.

Two things a map has that a spectrum does not:

**A step, and it matters.** The spatial step against the laser spot size
decides whether neighbouring pixels are independent measurements or the
same measurement twice. A 0.2 µm step with a 1 µm spot oversamples by
five, and every statistic computed pixel-wise is then correlated in a way
that makes a standard deviation meaningless. The map records the step and
says so.

**Missing pixels.** Line scans get aborted, and grids come back ragged.
A missing pixel is stored as NaN and every operation propagates it rather
than silently averaging it in as a zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ..core.spectrum import Spectrum


class MapError(ValueError):
    """Raised when a map cannot be built or used as asked."""


@dataclass
class RamanMap:
    """A grid of spectra sharing one spectral axis."""

    shift: np.ndarray
    """The common Raman shift axis, in cm⁻¹, length ``n_shift``."""
    intensity: np.ndarray
    """``(n_y, n_x, n_shift)``. NaN where a pixel is missing."""
    x: np.ndarray
    """Stage positions along the fast axis, in µm, length ``n_x``."""
    y: np.ndarray
    """Stage positions along the slow axis, in µm, length ``n_y``."""
    laser_nm: Optional[float] = None
    spot_um: Optional[float] = None
    """Laser spot diameter. Needed to say whether the sampling is
    independent; not guessed when absent."""
    name: str = "mapa"
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.shift = np.asarray(self.shift, dtype=float)
        self.intensity = np.asarray(self.intensity, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.intensity.ndim != 3:
            raise MapError(
                f"un mapa es un cubo (y, x, desplazamiento) y este tiene "
                f"{self.intensity.ndim} dimensiones"
            )
        expected = (self.y.size, self.x.size, self.shift.size)
        if self.intensity.shape != expected:
            raise MapError(
                f"el cubo es {self.intensity.shape} y los ejes dicen {expected}"
            )
        if self.shift.size < 2:
            raise MapError("hacen falta al menos dos puntos espectrales")
        if np.any(np.diff(self.shift) <= 0):
            order = np.argsort(self.shift)
            self.shift = self.shift[order]
            self.intensity = self.intensity[:, :, order]

    # -- shape ---------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.intensity.shape        # type: ignore[return-value]

    @property
    def n_pixels(self) -> int:
        return int(self.intensity.shape[0] * self.intensity.shape[1])

    @property
    def step_um(self) -> tuple[Optional[float], Optional[float]]:
        """The spatial step along each axis, or ``None`` for a single line."""
        return (_step(self.x), _step(self.y))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(x0, x1, y0, y1)`` for an image, in µm."""
        dx, dy = self.step_um
        half_x, half_y = (dx or 1.0) / 2, (dy or 1.0) / 2
        return (float(self.x[0] - half_x), float(self.x[-1] + half_x),
                float(self.y[0] - half_y), float(self.y[-1] + half_y))

    @property
    def missing(self) -> np.ndarray:
        """``(n_y, n_x)`` boolean: pixels with no spectrum."""
        return np.all(~np.isfinite(self.intensity), axis=2)

    def sampling_warning(self) -> Optional[str]:
        """Whether neighbouring pixels are independent measurements.

        Returns a warning when the step is smaller than half the spot: at
        that point adjacent pixels share most of their illuminated volume,
        and any pixel-wise statistic — a standard deviation, a histogram
        width, a cluster count — is measuring the optics.
        """
        if not self.spot_um:
            return ("el tamaño del punto láser no está declarado: no se puede "
                    "decir si los píxeles vecinos son medidas independientes")
        steps = [s for s in self.step_um if s]
        if not steps:
            return None
        smallest = min(steps)
        if smallest < self.spot_um / 2:
            return (
                f"el paso ({smallest:.2f} µm) es menor que medio punto láser "
                f"({self.spot_um / 2:.2f} µm): los píxeles vecinos miden en "
                "buena parte el mismo volumen, así que cualquier estadística "
                "píxel a píxel está midiendo la óptica"
            )
        return None

    # -- getting spectra out -------------------------------------------
    def spectrum_at(self, row: int, column: int) -> Spectrum:
        """One pixel, as a normal :class:`Spectrum`."""
        values = self.intensity[row, column]
        if not np.any(np.isfinite(values)):
            raise MapError(f"el píxel ({row}, {column}) no tiene espectro")
        return Spectrum(
            shift=self.shift, intensity=values, laser_nm=self.laser_nm,
            name=f"{self.name}[{row},{column}]",
            metadata={"mapa": self.name, "fila": row, "columna": column,
                      "x_um": float(self.x[column]), "y_um": float(self.y[row])},
        )

    def nearest(self, x_um: float, y_um: float) -> tuple[int, int]:
        """The pixel closest to a stage position."""
        return (int(np.argmin(np.abs(self.y - y_um))),
                int(np.argmin(np.abs(self.x - x_um))))

    def mean_spectrum(self) -> Spectrum:
        """The average over every pixel that has data.

        The one number a map always gives you, and the one that hides
        everything a map is for. It is here because it is the right
        reference for the rest, not because it is the answer.
        """
        import warnings

        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            mean = np.nanmean(self.intensity.reshape(-1, self.shift.size), axis=0)
        return Spectrum(shift=self.shift, intensity=mean, laser_nm=self.laser_nm,
                        name=f"{self.name} (media)",
                        metadata={"pixeles": self.n_pixels})

    def flat(self, drop_missing: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """The cube as ``(n_spectra, n_shift)``, with the pixel indices.

        Returns ``(matrix, indices)`` where ``indices`` is ``(n, 2)`` of
        ``(row, column)``, so a result computed on the matrix can be put
        back on the grid.
        """
        rows, columns, points = self.intensity.shape
        matrix = self.intensity.reshape(rows * columns, points)
        indices = np.stack(np.unravel_index(np.arange(rows * columns),
                                            (rows, columns)), axis=1)
        if drop_missing:
            keep = np.any(np.isfinite(matrix), axis=1)
            return matrix[keep], indices[keep]
        return matrix, indices

    def unflatten(self, values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Put a per-spectrum result back on the grid, NaN where missing."""
        rows, columns = self.intensity.shape[:2]
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            image = np.full((rows, columns), np.nan)
            image[indices[:, 0], indices[:, 1]] = values
            return image
        image = np.full((rows, columns, values.shape[1]), np.nan)
        image[indices[:, 0], indices[:, 1]] = values
        return image

    # -- reshaping -----------------------------------------------------
    def crop(self, low: float, high: float) -> "RamanMap":
        """The same map over a narrower spectral window."""
        keep = (self.shift >= low) & (self.shift <= high)
        if np.count_nonzero(keep) < 2:
            raise MapError(f"la ventana {low}–{high} cm⁻¹ deja menos de dos puntos")
        return self._replace(shift=self.shift[keep],
                             intensity=self.intensity[:, :, keep],
                             step=f"crop({low}, {high})")

    def region(self, rows: slice, columns: slice) -> "RamanMap":
        """A rectangular area of the map."""
        return RamanMap(
            shift=self.shift, intensity=self.intensity[rows, columns],
            x=self.x[columns], y=self.y[rows], laser_nm=self.laser_nm,
            spot_um=self.spot_um, name=f"{self.name} (recorte)",
            metadata=dict(self.metadata), history=list(self.history) + ["region"],
        )

    def with_intensity(self, intensity: np.ndarray, step: str) -> "RamanMap":
        """The same map with new values, recording what was done."""
        return self._replace(shift=self.shift, intensity=intensity, step=step)

    def apply(self, function, step: str = "apply") -> "RamanMap":
        """Run a per-spectrum function over every pixel.

        ``function`` takes and returns a :class:`Spectrum`. This is the
        slow path — it is a Python call per pixel — and it exists for the
        operations that cannot be vectorised, like a baseline fit.
        """
        rows, columns, _ = self.intensity.shape
        first: Optional[Spectrum] = None
        out: Optional[np.ndarray] = None
        for row in range(rows):
            for column in range(columns):
                if not np.any(np.isfinite(self.intensity[row, column])):
                    continue
                result = function(self.spectrum_at(row, column))
                if first is None:
                    first = result
                    out = np.full((rows, columns, result.shift.size), np.nan)
                elif result.shift.size != first.shift.size:
                    raise MapError(
                        "la función devuelve ejes de distinta longitud: un "
                        "mapa necesita un eje espectral común"
                    )
                out[row, column] = result.intensity        # type: ignore[index]
        if first is None or out is None:
            raise MapError("el mapa no tiene ningún píxel con datos")
        return RamanMap(
            shift=first.shift, intensity=out, x=self.x, y=self.y,
            laser_nm=self.laser_nm, spot_um=self.spot_um, name=self.name,
            metadata=dict(self.metadata), history=list(self.history) + [step],
        )

    def _replace(self, shift, intensity, step: str) -> "RamanMap":
        return RamanMap(
            shift=shift, intensity=intensity, x=self.x, y=self.y,
            laser_nm=self.laser_nm, spot_um=self.spot_um, name=self.name,
            metadata=dict(self.metadata), history=list(self.history) + [step],
        )

    def describe(self) -> str:
        rows, columns, points = self.intensity.shape
        dx, dy = self.step_um
        missing = int(np.count_nonzero(self.missing))
        text = (f"{self.name}: {rows} × {columns} píxeles, {points} puntos "
                f"espectrales ({self.shift[0]:.0f}–{self.shift[-1]:.0f} cm⁻¹)")
        if dx:
            text += f", paso {dx:.2f}"
            if dy and abs(dy - dx) > 1e-9:
                text += f" × {dy:.2f}"
            text += " µm"
        if missing:
            text += f", {missing} píxeles sin datos"
        return text

    def __repr__(self) -> str:                # pragma: no cover - debugging aid
        return f"<RamanMap {self.describe()}>"


def _step(axis: np.ndarray) -> Optional[float]:
    if axis.size < 2:
        return None
    steps = np.diff(axis)
    return float(np.median(np.abs(steps)))


def from_spectra(
    spectra: Sequence[Spectrum],
    positions: Sequence[tuple[float, float]],
    name: str = "mapa",
    spot_um: Optional[float] = None,
    tolerance: float = 1e-6,
) -> RamanMap:
    """Build a map from spectra with stage coordinates.

    The grid is deduced from the unique coordinates, and positions that do
    not fall on a grid are refused rather than snapped: a snapped position
    puts a spectrum in the wrong pixel, and the picture still looks fine.
    """
    if len(spectra) != len(positions):
        raise MapError(
            f"hay {len(spectra)} espectros y {len(positions)} posiciones")
    if not spectra:
        raise MapError("no hay espectros")

    axis = np.asarray(spectra[0].shift, dtype=float)
    for item in spectra[1:]:
        if item.shift.size != axis.size or not np.allclose(item.shift, axis,
                                                           rtol=0, atol=tolerance):
            raise MapError(
                "los espectros no comparten el eje espectral; interpólalos "
                "a un eje común antes de montar el mapa"
            )
    lasers = {s.laser_nm for s in spectra}
    if len(lasers) > 1:
        raise MapError("hay más de una longitud de onda de excitación en el mapa")

    xs = np.asarray([p[0] for p in positions], dtype=float)
    ys = np.asarray([p[1] for p in positions], dtype=float)
    unique_x = _unique(xs, tolerance)
    unique_y = _unique(ys, tolerance)

    filled = unique_y.size * unique_x.size
    if len(spectra) < 0.6 * filled:
        raise MapError(
            f"{len(spectra)} espectros ocuparían una rejilla de "
            f"{unique_y.size}×{unique_x.size} = {filled} celdas: las "
            "posiciones no caen en una rejilla, y ajustarlas a una pondría "
            "espectros en el píxel equivocado sin que la imagen lo delate"
        )

    cube = np.full((unique_y.size, unique_x.size, axis.size), np.nan)
    for spectrum, px, py in zip(spectra, xs, ys):
        column = int(np.argmin(np.abs(unique_x - px)))
        row = int(np.argmin(np.abs(unique_y - py)))
        if abs(unique_x[column] - px) > tolerance * 10 or \
                abs(unique_y[row] - py) > tolerance * 10:
            raise MapError(
                f"la posición ({px}, {py}) no cae en la rejilla; un mapa "
                "irregular hay que interpolarlo explícitamente"
            )
        cube[row, column] = spectrum.intensity
    return RamanMap(
        shift=axis, intensity=cube, x=unique_x, y=unique_y,
        laser_nm=spectra[0].laser_nm, spot_um=spot_um, name=name,
        metadata={"espectros": len(spectra)},
    )


def _unique(values: np.ndarray, tolerance: float) -> np.ndarray:
    """Distinct coordinates, merging ones closer than the tolerance."""
    ordered = np.sort(np.unique(values))
    if ordered.size == 0:
        return ordered
    kept = [ordered[0]]
    for value in ordered[1:]:
        if value - kept[-1] > max(tolerance * 10, 1e-9):
            kept.append(value)
    return np.asarray(kept, dtype=float)


__all__ = ["MapError", "RamanMap", "from_spectra"]
