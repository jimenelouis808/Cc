"""Unit transforms for secondary axes.

A diffraction pattern is read in 2θ and thought about in d-spacing; a
Raman spectrum is measured in shift and the laser is specified in
nanometres; an impedance spectrum is plotted against frequency and
interpreted as time constants. Drawing the second axis by hand means
getting a non-linear transform right, and then choosing tick positions
that are round numbers on the *secondary* scale rather than on the
primary — which is what makes the difference between a second axis that
helps and one that is decoration.

Each transform is a pair of functions and its own tick chooser. They are
exact inverses; there is a test that checks it, because a second axis that
disagrees with the first is worse than none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

#: Planck constant times c, in eV·nm. Used for shift ↔ energy.
HC_EV_NM = 1239.841984


@dataclass(frozen=True)
class Transform:
    """A named change of units for a secondary axis."""

    key: str
    label: str
    """Default axis label, with units."""
    forward: Callable[[np.ndarray, float], np.ndarray]
    """Primary → secondary."""
    inverse: Callable[[np.ndarray, float], np.ndarray]
    """Secondary → primary."""
    parameter_name: str = ""
    parameter_units: str = ""
    nice_ticks: Optional[Sequence[float]] = None
    """Preferred tick values in secondary units; the ones inside the
    visible range are used."""

    def __call__(self, values, parameter: float) -> np.ndarray:
        return self.forward(np.asarray(values, dtype=float), parameter)


def _two_theta_to_d(two_theta: np.ndarray, wavelength: float) -> np.ndarray:
    """Bragg: ``d = λ / (2 sin θ)``."""
    sine = np.sin(np.radians(np.asarray(two_theta, dtype=float)) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = wavelength / (2.0 * sine)
    return np.where(np.isfinite(out), out, np.inf)


def _d_to_two_theta(d: np.ndarray, wavelength: float) -> np.ndarray:
    ratio = wavelength / (2.0 * np.asarray(d, dtype=float))
    ratio = np.clip(ratio, -1.0, 1.0)
    return 2.0 * np.degrees(np.arcsin(ratio))


def _two_theta_to_q(two_theta: np.ndarray, wavelength: float) -> np.ndarray:
    """``q = 4π sin θ / λ``, in Å⁻¹."""
    sine = np.sin(np.radians(np.asarray(two_theta, dtype=float)) / 2.0)
    return 4.0 * math.pi * sine / wavelength


def _q_to_two_theta(q: np.ndarray, wavelength: float) -> np.ndarray:
    ratio = np.asarray(q, dtype=float) * wavelength / (4.0 * math.pi)
    return 2.0 * np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0)))


def _shift_to_wavelength(shift: np.ndarray, laser_nm: float) -> np.ndarray:
    """Raman shift in cm⁻¹ to absolute (Stokes) wavelength in nm.

    ``1/λ_scattered = 1/λ_laser − Δν̃``, with the shift in cm⁻¹ and the
    wavelengths converted to cm. Anti-Stokes shifts (negative) come out
    below the laser line, which is correct.
    """
    inverse_laser = 1e7 / laser_nm          # cm-1
    scattered = inverse_laser - np.asarray(shift, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1e7 / scattered
    return np.where(np.isfinite(out), out, np.nan)


def _wavelength_to_shift(wavelength: np.ndarray, laser_nm: float) -> np.ndarray:
    return 1e7 / laser_nm - 1e7 / np.asarray(wavelength, dtype=float)


def _shift_to_mev(shift: np.ndarray, _: float = 0.0) -> np.ndarray:
    """cm⁻¹ to meV: ``1 cm⁻¹ = 0.1239842 meV``."""
    return np.asarray(shift, dtype=float) * 0.1239841984


def _mev_to_shift(mev: np.ndarray, _: float = 0.0) -> np.ndarray:
    return np.asarray(mev, dtype=float) / 0.1239841984


def _offset(values: np.ndarray, offset: float) -> np.ndarray:
    """Add a constant. Used for potential against a second reference."""
    return np.asarray(values, dtype=float) + offset


def _unoffset(values: np.ndarray, offset: float) -> np.ndarray:
    return np.asarray(values, dtype=float) - offset


def _frequency_to_tau(frequency: np.ndarray, _: float = 0.0) -> np.ndarray:
    """Hz to the time constant ``τ = 1/(2πf)`` in seconds."""
    values = np.asarray(frequency, dtype=float)
    with np.errstate(divide="ignore"):
        out = 1.0 / (2.0 * math.pi * values)
    return np.where(np.isfinite(out), out, np.inf)


def _tau_to_frequency(tau: np.ndarray, _: float = 0.0) -> np.ndarray:
    values = np.asarray(tau, dtype=float)
    with np.errstate(divide="ignore"):
        out = 1.0 / (2.0 * math.pi * values)
    return np.where(np.isfinite(out), out, np.inf)


def _wavenumber_to_micron(wavenumber: np.ndarray, _: float = 0.0) -> np.ndarray:
    values = np.asarray(wavenumber, dtype=float)
    with np.errstate(divide="ignore"):
        out = 1e4 / values
    return np.where(np.isfinite(out), out, np.inf)


#: Every available secondary-axis transform.
TRANSFORMS: dict[str, Transform] = {
    "d": Transform(
        key="d", label="d (Å)",
        forward=_two_theta_to_d, inverse=_d_to_two_theta,
        parameter_name="longitud de onda", parameter_units="Å",
        nice_ticks=(10, 8, 6, 5, 4, 3.5, 3, 2.5, 2.2, 2, 1.8, 1.6, 1.5,
                    1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8),
    ),
    "q": Transform(
        key="q", label="q (Å⁻¹)",
        forward=_two_theta_to_q, inverse=_q_to_two_theta,
        parameter_name="longitud de onda", parameter_units="Å",
        nice_ticks=tuple(np.arange(0.5, 9.01, 0.5)),
    ),
    "lambda": Transform(
        key="lambda", label="Longitud de onda (nm)",
        forward=_shift_to_wavelength, inverse=_wavelength_to_shift,
        parameter_name="láser", parameter_units="nm",
    ),
    "mev": Transform(
        key="mev", label="Energía (meV)",
        forward=_shift_to_mev, inverse=_mev_to_shift,
    ),
    "micron": Transform(
        key="micron", label="Longitud de onda (µm)",
        forward=_wavenumber_to_micron, inverse=_wavenumber_to_micron,
    ),
    "referencia": Transform(
        key="referencia", label="Potencial (V)",
        forward=_offset, inverse=_unoffset,
        parameter_name="desplazamiento", parameter_units="V",
    ),
    "tau": Transform(
        key="tau", label="τ (s)",
        forward=_frequency_to_tau, inverse=_tau_to_frequency,
    ),
}


def secondary_ticks(
    transform: Transform,
    primary_limits: tuple[float, float],
    parameter: float,
    count: int = 6,
    explicit: Optional[Sequence[float]] = None,
) -> tuple[list[float], list[float]]:
    """Tick positions for a secondary axis.

    Returns ``(positions in PRIMARY units, labels in SECONDARY units)``.

    The values are round on the *secondary* scale, which is the whole
    point: a d-spacing axis with ticks at 3.113, 2.668 and 2.403 Å because
    those are where the round 2θ values fell is not a d-spacing axis
    anybody can read.

    Ticks that would land on top of each other are dropped. A non-linear
    transform bunches its round values at one end — the d-spacing list
    crowds ten labels into the last fifth of a diffraction axis — and in a
    panel figure they overprint into a smear. Only the automatic tick
    lists are thinned; an explicit list is drawn as asked.
    """
    low, high = float(min(primary_limits)), float(max(primary_limits))
    ends = transform.forward(np.array([low, high]), parameter)
    finite = ends[np.isfinite(ends)]
    if finite.size < 2:
        return [], []
    lo, hi = float(finite.min()), float(finite.max())

    if explicit is not None:
        candidates = [float(v) for v in explicit]
    elif transform.nice_ticks:
        candidates = [float(v) for v in transform.nice_ticks]
    else:
        candidates = list(_round_values(lo, hi, count))

    minimum_gap = 0.0 if explicit is not None else (high - low) / (count * 1.6)

    positions: list[float] = []
    labels: list[float] = []
    for value in candidates:
        if not lo <= value <= hi:
            continue
        primary = float(transform.inverse(np.array([value]), parameter)[0])
        if not math.isfinite(primary) or not low <= primary <= high:
            continue
        if positions and abs(primary - positions[-1]) < minimum_gap:
            continue
        positions.append(primary)
        labels.append(value)
    return positions, labels


def _round_values(low: float, high: float, count: int) -> list[float]:
    """Round numbers spanning a range, at a 1/2/5 step."""
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return []
    raw = (high - low) / max(count, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiple * magnitude
        if (high - low) / step <= count * 1.4:
            break
    start = math.ceil(low / step) * step
    values = []
    value = start
    while value <= high + 1e-9:
        values.append(round(value, 10))
        value += step
    return values


__all__ = ["HC_EV_NM", "TRANSFORMS", "Transform", "secondary_ticks"]
