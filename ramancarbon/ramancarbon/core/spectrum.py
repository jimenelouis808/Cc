"""The :class:`Spectrum` container and the operations that keep it honest.

A Raman spectrum here is a pair of equal-length arrays — Raman shift in
cm⁻¹ and intensity in whatever arbitrary units the instrument produced —
plus the metadata that decides what the numbers *mean*:

* ``laser_nm`` — the excitation wavelength. Nothing in carbon Raman can be
  interpreted without it. The D band sits at 1350 cm⁻¹ at 532 nm and near
  1330 cm⁻¹ at 785 nm, and the 2D band moves twice as fast. Ratios such as
  I_D/I_G are only comparable between spectra taken at the same laser.
* ``history`` — every preprocessing step applied, in order. A baseline
  subtraction or a normalisation changes intensity ratios, so a fitted
  I_D/I_G is meaningless without knowing what was done to get there.

Intensities are deliberately *not* given units. Raman intensity is a
relative quantity; this package never pretends otherwise.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional, Sequence

import numpy as np

#: Planck constant times the speed of light, in eV·nm. Converts a laser
#: wavelength to a photon energy: E(eV) = 1239.841984 / λ(nm).
HC_EV_NM = 1239.841984


def laser_energy_ev(wavelength_nm: float) -> float:
    """Photon energy of a laser line, in eV.

    Parameters
    ----------
    wavelength_nm:
        Excitation wavelength in nanometres (532, 633, 785…).

    Returns
    -------
    float
        Photon energy in eV.

    Raises
    ------
    ValueError
        If the wavelength is not positive.
    """
    if wavelength_nm <= 0:
        raise ValueError("laser wavelength must be positive, got %r" % (wavelength_nm,))
    return HC_EV_NM / float(wavelength_nm)


def laser_wavelength_nm(energy_ev: float) -> float:
    """Inverse of :func:`laser_energy_ev`: eV back to nanometres."""
    if energy_ev <= 0:
        raise ValueError("laser energy must be positive, got %r" % (energy_ev,))
    return HC_EV_NM / float(energy_ev)


@dataclass
class Spectrum:
    """One Raman spectrum and everything needed to interpret it.

    Parameters
    ----------
    shift:
        Raman shift axis in cm⁻¹. Stored sorted ascending; duplicated
        abscissae are averaged, because several instruments emit them at
        stitch boundaries between grating windows and they break every
        interpolation downstream.
    intensity:
        Intensity in arbitrary units, same length as ``shift``.
    laser_nm:
        Excitation wavelength in nm. ``None`` means unknown, and the
        analysis modules will refuse the calculations that need it rather
        than silently assuming 532 nm.
    name:
        Human-readable label, normally the file stem.
    metadata:
        Free-form dictionary of whatever the reader could recover
        (integration time, objective, grating, instrument…).
    history:
        Ordered log of preprocessing steps applied to this spectrum.
    """

    shift: np.ndarray
    intensity: np.ndarray
    laser_nm: Optional[float] = None
    name: str = "spectrum"
    metadata: dict = field(default_factory=dict)
    history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.shift = np.asarray(self.shift, dtype=float).ravel()
        self.intensity = np.asarray(self.intensity, dtype=float).ravel()
        if self.shift.size != self.intensity.size:
            raise ValueError(
                "shift and intensity must have the same length "
                f"({self.shift.size} vs {self.intensity.size})"
            )
        if self.shift.size < 2:
            raise ValueError("a spectrum needs at least two points")
        if not np.all(np.isfinite(self.shift)):
            raise ValueError("the Raman shift axis contains NaN or inf")
        self._normalise_axis()
        if self.laser_nm is not None:
            laser_energy_ev(self.laser_nm)  # validates positivity

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _normalise_axis(self) -> None:
        """Sort ascending and average duplicate abscissae."""
        order = np.argsort(self.shift, kind="stable")
        x = self.shift[order]
        y = self.intensity[order]
        # Collapse exact duplicates by averaging their intensities.
        unique_x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        if unique_x.size != x.size:
            summed = np.zeros(unique_x.size, dtype=float)
            np.add.at(summed, inverse, y)
            y = summed / counts
            x = unique_x
        self.shift = x
        self.intensity = y

    def copy(self) -> "Spectrum":
        """Deep copy, including metadata and history."""
        return Spectrum(
            shift=self.shift.copy(),
            intensity=self.intensity.copy(),
            laser_nm=self.laser_nm,
            name=self.name,
            metadata=copy.deepcopy(self.metadata),
            history=list(self.history),
        )

    def with_intensity(self, intensity: Sequence[float], step: str) -> "Spectrum":
        """Return a copy carrying new intensities and one more history entry.

        Every preprocessing routine goes through here, so the history is
        complete by construction rather than by discipline.
        """
        out = self.copy()
        new = np.asarray(intensity, dtype=float).ravel()
        if new.size != out.shift.size:
            raise ValueError(
                f"new intensity has {new.size} points, axis has {out.shift.size}"
            )
        out.intensity = new
        out.history.append(step)
        return out

    # ------------------------------------------------------------------
    # basic properties
    # ------------------------------------------------------------------
    @property
    def laser_ev(self) -> Optional[float]:
        """Excitation photon energy in eV, or ``None`` if the laser is unknown."""
        if self.laser_nm is None:
            return None
        return laser_energy_ev(self.laser_nm)

    @property
    def range(self) -> tuple[float, float]:
        """(min, max) of the Raman shift axis, in cm⁻¹."""
        return float(self.shift[0]), float(self.shift[-1])

    @property
    def step(self) -> float:
        """Median spacing of the shift axis in cm⁻¹.

        Median rather than mean: stitched spectra have one huge gap at the
        window boundary that would drag a mean far from the real sampling.
        """
        return float(np.median(np.diff(self.shift)))

    @property
    def is_uniform(self) -> bool:
        """Whether the axis is evenly sampled to within 1 % of its step.

        Savitzky–Golay smoothing and FFT-based operations assume this;
        :func:`~ramancarbon.core.preprocess.resample` fixes it when it fails.
        """
        d = np.diff(self.shift)
        if d.size == 0:
            return True
        return bool(np.max(np.abs(d - np.median(d))) <= 0.01 * abs(np.median(d)))

    def covers(self, low: float, high: float, fraction: float = 0.9) -> bool:
        """Whether the axis spans at least ``fraction`` of ``[low, high]``.

        Used throughout the analysis to decide whether a band is *absent*
        or merely *outside the measured window* — a distinction that
        changes the classification of a sample entirely. An RBM-free
        spectrum starting at 400 cm⁻¹ says nothing about SWCNTs.
        """
        lo = max(low, self.shift[0])
        hi = min(high, self.shift[-1])
        if hi <= lo:
            return False
        return (hi - lo) >= fraction * (high - low)

    # ------------------------------------------------------------------
    # slicing and sampling
    # ------------------------------------------------------------------
    def crop(self, low: float, high: float) -> "Spectrum":
        """Restrict the spectrum to ``[low, high]`` cm⁻¹ (inclusive)."""
        if high <= low:
            raise ValueError(f"empty crop window [{low}, {high}]")
        mask = (self.shift >= low) & (self.shift <= high)
        if mask.sum() < 2:
            raise ValueError(
                f"window [{low}, {high}] cm⁻¹ contains {int(mask.sum())} points; "
                f"the spectrum spans {self.range[0]:.1f}–{self.range[1]:.1f} cm⁻¹"
            )
        out = self.copy()
        out.shift = self.shift[mask]
        out.intensity = self.intensity[mask]
        out.history.append(f"crop({low:g}, {high:g})")
        return out

    def region(self, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
        """Raw ``(x, y)`` arrays inside a window, without building a Spectrum.

        Cheaper than :meth:`crop` for the inner loops of peak search and
        integration, and does not raise on a short window — callers that
        need a guarantee should use :meth:`crop`.
        """
        mask = (self.shift >= low) & (self.shift <= high)
        return self.shift[mask], self.intensity[mask]

    def interpolate_at(self, x: Sequence[float]) -> np.ndarray:
        """Linear interpolation of the intensity at arbitrary shifts."""
        return np.interp(np.asarray(x, dtype=float), self.shift, self.intensity)

    # ------------------------------------------------------------------
    # measurements
    # ------------------------------------------------------------------
    def max_in(self, low: float, high: float) -> Optional[tuple[float, float]]:
        """``(position, intensity)`` of the tallest point in a window.

        Returns ``None`` when the window is empty. This is a raw maximum,
        not a fitted peak position — use it for quick looks and seeds, not
        for reported band positions.
        """
        x, y = self.region(low, high)
        if x.size == 0:
            return None
        i = int(np.argmax(y))
        return float(x[i]), float(y[i])

    def area_in(self, low: float, high: float) -> float:
        """Trapezoidal integral of the intensity over a window.

        No baseline is removed here. Integrating a spectrum that still
        carries fluorescence gives an area dominated by the background, so
        baseline-correct first.
        """
        x, y = self.region(low, high)
        if x.size < 2:
            return 0.0
        return float(np.trapezoid(y, x))

    def noise_estimate(self, low: Optional[float] = None, high: Optional[float] = None) -> float:
        """Robust estimate of the noise standard deviation.

        Uses the median absolute deviation of the second difference, scaled
        by ``1 / (0.6745 * sqrt(6))``. The second difference kills any
        smooth background and any linear trend, so the estimate survives
        being run on a region that still contains a broad band — which is
        why it is preferred here over "the standard deviation of a flat
        region", a region that carbon spectra rarely have.

        Parameters
        ----------
        low, high:
            Optional window. Defaults to the whole spectrum.

        Returns
        -------
        float
            Estimated per-point noise σ in intensity units.
        """
        y = self.intensity if low is None or high is None else self.region(low, high)[1]
        if y.size < 5:
            return 0.0
        d2 = np.diff(y, n=2)
        mad = float(np.median(np.abs(d2 - np.median(d2))))
        return mad / (0.6745 * np.sqrt(6.0))

    def snr_at(self, position: float, window: float = 20.0) -> float:
        """Crude peak signal-to-noise: local height over the global noise σ.

        A band whose SNR is below ~3 should not be reported as detected,
        and this package's peak finder enforces that by default.
        """
        sigma = self.noise_estimate()
        if sigma <= 0:
            return float("inf")
        peak = self.max_in(position - window, position + window)
        if peak is None:
            return 0.0
        return peak[1] / sigma

    # ------------------------------------------------------------------
    def describe(self) -> str:
        """One-paragraph human summary, used by the CLI and the GUI header."""
        lo, hi = self.range
        laser = f"{self.laser_nm:g} nm ({self.laser_ev:.3f} eV)" if self.laser_nm else "desconocido"
        lines = [
            f"{self.name}: {self.shift.size} puntos, {lo:.1f}–{hi:.1f} cm⁻¹, "
            f"paso {self.step:.2f} cm⁻¹",
            f"láser: {laser}",
        ]
        if not self.is_uniform:
            lines.append("eje no uniforme (remuestrea antes de suavizar)")
        if self.history:
            lines.append("procesado: " + " → ".join(self.history))
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        lo, hi = self.range
        return (
            f"Spectrum(name={self.name!r}, n={self.shift.size}, "
            f"range=({lo:.1f}, {hi:.1f}), laser_nm={self.laser_nm})"
        )


def stack_average(spectra: Iterable[Spectrum], name: str = "promedio") -> Spectrum:
    """Average several spectra onto the axis of the first one.

    Accumulation improves signal-to-noise as sqrt(N) only if the spectra
    really are repeats of the same measurement. Mixing lasers is a physics
    error, not a bookkeeping one, so it raises.

    Parameters
    ----------
    spectra:
        Two or more spectra. All must share the same ``laser_nm`` (or all
        leave it ``None``).
    name:
        Name for the result.

    Returns
    -------
    Spectrum
        The point-wise mean, interpolated onto the first spectrum's axis
        and restricted to the overlap of all inputs.
    """
    items = list(spectra)
    if not items:
        raise ValueError("nothing to average")
    if len(items) == 1:
        return items[0].copy()
    lasers = {s.laser_nm for s in items}
    if len(lasers) > 1:
        raise ValueError(
            "refusing to average spectra taken at different excitations: "
            + ", ".join(repr(v) for v in sorted(lasers, key=lambda v: (v is None, v)))
        )
    lo = max(s.shift[0] for s in items)
    hi = min(s.shift[-1] for s in items)
    if hi <= lo:
        raise ValueError("the spectra do not overlap in Raman shift")
    base = items[0]
    axis = base.shift[(base.shift >= lo) & (base.shift <= hi)]
    if axis.size < 2:
        raise ValueError("the overlap window is too narrow to average over")
    acc = np.mean([s.interpolate_at(axis) for s in items], axis=0)
    return Spectrum(
        shift=axis,
        intensity=acc,
        laser_nm=base.laser_nm,
        name=name,
        metadata={"averaged_from": [s.name for s in items]},
        history=[f"average(n={len(items)})"],
    )


__all__ = [
    "HC_EV_NM",
    "Spectrum",
    "laser_energy_ev",
    "laser_wavelength_nm",
    "stack_average",
]
