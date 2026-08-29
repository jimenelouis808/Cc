"""Parsing and plotting vibrational spectra from ``dynmat.x``.

``dynmat.x`` prints a table of modes to standard output, which the runner
script written by :mod:`carbonforge.exports.qe` captures as ``dynmat.out``::

    # mode   [cm-1]    [THz]      IR          Raman   depol.fact
       1      0.00     0.0000    0.0000      0.0000    0.0000
       2    865.31    25.9414    0.0132     18.4471    0.7500

The Raman and depolarisation columns only appear when ``ph.x`` ran with
``lraman``; the IR column only when it ran with ``epsil``. The parser
handles all three shapes.

Units, since they are easy to misreport: frequencies in cm⁻¹, IR activity in
(D/Å)²/amu, Raman activity in Å⁴/amu. These are **activities**, not
experimental intensities: converting to what a spectrometer sees additionally
requires the Bose occupation factor and the (ω_laser − ω)⁴ scattering
prefactor. :func:`broaden` can apply both, and says so, rather than quietly
plotting activities on an axis labelled "intensity".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

#: Physical constants in the units used here.
_CM1_TO_K = 1.4387768775039337  # hc/k_B, converts cm^-1 to kelvin


@dataclass
class VibrationalMode:
    """One normal mode."""

    index: int
    frequency_cm1: float
    frequency_thz: Optional[float] = None
    ir_activity: Optional[float] = None
    raman_activity: Optional[float] = None
    depolarisation: Optional[float] = None

    @property
    def is_acoustic(self) -> bool:
        """Whether this looks like one of the three translational modes.

        Uses a 10 cm⁻¹ window around zero. With the acoustic sum rule applied
        these come out at exactly zero; without it they land at tens of cm⁻¹
        and this heuristic stops working — which is itself a signal that the
        sum rule was skipped.
        """
        return abs(self.frequency_cm1) < 10.0

    @property
    def is_imaginary(self) -> bool:
        """Whether the mode is unstable.

        ``dynmat.x`` reports imaginary frequencies as negative numbers. A
        genuinely negative mode (beyond the acoustic noise window) means the
        structure is at a saddle point, not a minimum: relax it further
        before trusting any spectrum computed from it.
        """
        return self.frequency_cm1 < -10.0


@dataclass
class VibrationalSpectrum:
    """A parsed set of normal modes."""

    modes: list[VibrationalMode] = field(default_factory=list)
    has_ir: bool = False
    has_raman: bool = False

    def __len__(self) -> int:
        return len(self.modes)

    @property
    def frequencies(self) -> np.ndarray:
        """All frequencies in cm⁻¹."""
        return np.array([m.frequency_cm1 for m in self.modes])

    @property
    def imaginary_modes(self) -> list[VibrationalMode]:
        """Unstable modes, which invalidate the spectrum if present."""
        return [m for m in self.modes if m.is_imaginary]

    def optical_modes(self) -> list[VibrationalMode]:
        """Modes excluding the acoustic branch."""
        return [m for m in self.modes if not m.is_acoustic]

    def activities(self, kind: str) -> np.ndarray:
        """Return the activity column for ``kind`` (``"ir"`` or ``"raman"``).

        Missing values become zero, so a mode that carries no activity simply
        does not contribute to the spectrum.
        """
        if kind == "ir":
            if not self.has_ir:
                raise ValueError(
                    "Este cálculo no incluye actividades IR. Hace falta "
                    "epsil=.true. en ph.x."
                )
            return np.array([m.ir_activity or 0.0 for m in self.modes])
        if kind == "raman":
            if not self.has_raman:
                raise ValueError(
                    "Este cálculo no incluye actividades Raman. Hace falta "
                    "lraman=.true. en ph.x (y pseudos norm-conserving)."
                )
            return np.array([m.raman_activity or 0.0 for m in self.modes])
        raise ValueError(f"kind debe ser 'ir' o 'raman', no {kind!r}.")

    def summary(self) -> str:
        """Short human-readable report, including stability warnings."""
        lines = [f"{len(self.modes)} modos normales."]
        optical = self.optical_modes()
        if optical:
            freqs = np.array([m.frequency_cm1 for m in optical])
            lines.append(
                f"Rango óptico: {freqs.min():.1f} – {freqs.max():.1f} cm⁻¹"
            )
        lines.append(f"Actividades IR: {'sí' if self.has_ir else 'no'}")
        lines.append(f"Actividades Raman: {'sí' if self.has_raman else 'no'}")

        imaginary = self.imaginary_modes
        if imaginary:
            worst = min(m.frequency_cm1 for m in imaginary)
            lines.append(
                f"\n⚠️  {len(imaginary)} modo(s) imaginario(s), el más negativo "
                f"a {worst:.1f} cm⁻¹. La estructura NO está en un mínimo: "
                "relájala mejor antes de fiarte del espectro."
            )
        n_acoustic = sum(1 for m in self.modes if m.is_acoustic)
        if n_acoustic != 3 and len(self.modes) > 3:
            lines.append(
                f"\n⚠️  Se esperaban 3 modos acústicos cerca de 0 cm⁻¹ y hay "
                f"{n_acoustic}. Suele indicar que no se aplicó la regla de "
                "suma acústica (asr) o que la relajación es insuficiente."
            )
        return "\n".join(lines)


# Matches a data row: an integer index followed by at least one float.
_ROW = re.compile(
    r"^\s*(\d+)\s+(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)"
    r"((?:\s+-?\d+\.\d+(?:[eEdD][+-]?\d+)?)*)\s*$"
)


def read_dynmat(path: str | Path) -> VibrationalSpectrum:
    """Parse the mode table from ``dynmat.x`` standard output.

    Parameters
    ----------
    path
        Path to the captured stdout (``dynmat.out`` in the generated runner
        script).

    Returns
    -------
    VibrationalSpectrum

    Raises
    ------
    ValueError
        When no mode table can be found, which usually means ``dynmat.x``
        stopped early — check the file for its error message.
    """
    text = Path(path).read_text()
    lines = text.splitlines()

    # Locate the header so we know which optional columns are present.
    has_ir = False
    has_raman = False
    header_index = None
    for index, line in enumerate(lines):
        lowered = line.lower()
        if "mode" in lowered and "cm-1" in lowered:
            header_index = index
            has_ir = "ir" in lowered
            has_raman = "raman" in lowered
            break

    search_region = lines[header_index + 1:] if header_index is not None else lines

    modes: list[VibrationalMode] = []
    for line in search_region:
        match = _ROW.match(line)
        if not match:
            # Stop at the first non-row once the table has begun; keep
            # scanning while looking for its start.
            if modes:
                break
            continue
        index = int(match.group(1))
        numbers = [float(match.group(2).replace("D", "E").replace("d", "e"))]
        rest = match.group(3).split()
        numbers.extend(float(v.replace("D", "E").replace("d", "e")) for v in rest)

        mode = VibrationalMode(index=index, frequency_cm1=numbers[0])
        # Columns after the header are, in order:
        #   [THz]  then optionally IR, then optionally Raman + depol.
        cursor = 1
        if len(numbers) > cursor:
            mode.frequency_thz = numbers[cursor]
            cursor += 1
        if has_ir and len(numbers) > cursor:
            mode.ir_activity = numbers[cursor]
            cursor += 1
        if has_raman and len(numbers) > cursor:
            mode.raman_activity = numbers[cursor]
            cursor += 1
        if has_raman and len(numbers) > cursor:
            mode.depolarisation = numbers[cursor]
        modes.append(mode)

    if not modes:
        raise ValueError(
            f"{path}: no se encontró la tabla de modos. ¿Se detuvo dynmat.x? "
            "Revisa el archivo, que suele contener el mensaje de error."
        )

    return VibrationalSpectrum(
        modes=modes,
        has_ir=has_ir and any(m.ir_activity is not None for m in modes),
        has_raman=has_raman and any(m.raman_activity is not None for m in modes),
    )


def broaden(
    frequencies: Sequence[float],
    activities: Sequence[float],
    width_cm1: float = 8.0,
    grid: Optional[np.ndarray] = None,
    padding_cm1: float = 100.0,
    n_points: int = 2000,
    laser_wavelength_nm: Optional[float] = None,
    temperature_k: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convolve discrete lines into a continuous spectrum with Lorentzians.

    Parameters
    ----------
    frequencies, activities
        Mode positions (cm⁻¹) and their activities. Equal length.
    width_cm1
        Half-width at half-maximum. 4-10 cm⁻¹ matches typical Raman
        instrumental resolution.
    grid
        Explicit frequency grid. Built automatically when omitted.
    padding_cm1, n_points
        Control the automatic grid.
    laser_wavelength_nm
        When given, apply the ``(ν_laser − ν)⁴`` scattering prefactor that
        converts a Raman *activity* into something proportional to measured
        intensity.
    temperature_k
        When given, apply the Bose-Einstein occupation factor
        ``1 / (1 − exp(−hcν / k_BT))`` for Stokes scattering.

    Returns
    -------
    (grid, intensity)
        Both 1-D arrays of length ``n_points``.

    Notes
    -----
    Passing neither ``laser_wavelength_nm`` nor ``temperature_k`` returns the
    broadened raw activities. That is the honest default: it is what the
    calculation produced. The two corrections are what make a computed
    spectrum comparable to an experimental one, and are opt-in so it is
    always clear which is being shown.
    """
    freqs = np.asarray(frequencies, dtype=float)
    acts = np.asarray(activities, dtype=float)
    if freqs.shape != acts.shape:
        raise ValueError(
            f"frequencies y activities deben tener la misma longitud "
            f"({freqs.shape} vs {acts.shape})."
        )
    if width_cm1 <= 0:
        raise ValueError("width_cm1 debe ser positivo.")

    # Only positive-frequency modes carry Stokes intensity; acoustic and
    # imaginary modes are excluded rather than producing a spurious peak at 0.
    keep = freqs > 10.0
    freqs, acts = freqs[keep], acts[keep]

    if grid is None:
        if freqs.size == 0:
            grid = np.linspace(0.0, padding_cm1, n_points)
        else:
            grid = np.linspace(
                max(0.0, float(freqs.min()) - padding_cm1),
                float(freqs.max()) + padding_cm1,
                n_points,
            )
    grid = np.asarray(grid, dtype=float)

    weights = acts.astype(float).copy()
    if temperature_k is not None and freqs.size:
        if temperature_k <= 0:
            raise ValueError("temperature_k debe ser positiva.")
        # Stokes Bose factor: n(ν) + 1 = 1 / (1 - exp(-hcν/kT)).
        exponent = -_CM1_TO_K * freqs / temperature_k
        weights = weights / (1.0 - np.exp(exponent))
    if laser_wavelength_nm is not None and freqs.size:
        if laser_wavelength_nm <= 0:
            raise ValueError("laser_wavelength_nm debe ser positiva.")
        laser_cm1 = 1.0e7 / laser_wavelength_nm
        if np.any(freqs >= laser_cm1):
            raise ValueError(
                "Hay modos por encima de la frecuencia del láser; el factor "
                "(ν_láser − ν)⁴ no tiene sentido. Revisa la longitud de onda."
            )
        weights = weights * (laser_cm1 - freqs) ** 4 / laser_cm1 ** 4

    intensity = np.zeros_like(grid)
    for centre, weight in zip(freqs, weights):
        intensity += weight * (width_cm1 / np.pi) / (
            (grid - centre) ** 2 + width_cm1 ** 2
        )
    return grid, intensity


def plot_spectrum(
    spectrum: VibrationalSpectrum,
    kind: str = "raman",
    width_cm1: float = 8.0,
    laser_wavelength_nm: Optional[float] = None,
    temperature_k: Optional[float] = None,
    title: Optional[str] = None,
):
    """Return a matplotlib ``Figure`` with the broadened spectrum and its lines.

    Parameters
    ----------
    spectrum
        Parsed modes.
    kind
        ``"raman"`` or ``"ir"``.
    width_cm1
        Lorentzian half-width.
    laser_wavelength_nm, temperature_k
        Passed to :func:`broaden`. The axis label states whether corrections
        were applied.
    title
        Figure title.
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

    activities = spectrum.activities(kind)
    freqs = spectrum.frequencies
    grid, intensity = broaden(
        freqs, activities, width_cm1=width_cm1,
        laser_wavelength_nm=laser_wavelength_nm,
        temperature_k=temperature_k,
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(grid, intensity, color="#1f4e9c", linewidth=1.2)

    # Stick plot underneath, scaled to the broadened curve for readability.
    peak = intensity.max() if intensity.max() > 0 else 1.0
    stick_scale = peak / (activities.max() if activities.max() > 0 else 1.0)
    for frequency, activity in zip(freqs, activities):
        if frequency > 10.0 and activity > 0:
            ax.vlines(frequency, 0, activity * stick_scale,
                      color="#c0392b", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Número de onda (cm⁻¹)")
    corrected = laser_wavelength_nm is not None or temperature_k is not None
    label = "Intensidad (u.a.)" if corrected else "Actividad (u.a.)"
    ax.set_ylabel(label)
    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(bottom=0)
    ax.set_title(
        title or f"Espectro {'Raman' if kind == 'raman' else 'infrarrojo'}"
    )
    fig.tight_layout()
    return fig
