"""The XPS spectrum, and the two energy scales it lives on.

A photoelectron spectrum is measured in **kinetic** energy and read in
**binding** energy, and the conversion runs through the photon energy and
the spectrometer work function:

.. math::  E_B = h\\nu - E_K - \\phi

Everything downstream depends on which scale a number is on, and the two
are not interchangeable:

**Photoelectron lines sit at a fixed binding energy.** Change the anode
from Al to Mg and C 1s stays at 284.8 eV.

**Auger lines sit at a fixed kinetic energy.** The same change moves the
C KLL by 233 eV on the binding-energy axis, because that axis moved under
it. This is why a survey identifier that works in binding energy alone
assigns Auger peaks to whatever element happens to be there — and with
Al Kα the Ni LMM lands on top of Fe 2p.

So the spectrum carries its photon energy, and it is not optional. A file
that does not state it is read, and every analysis that needs the kinetic
scale then refuses with that as the reason rather than assuming aluminium.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

#: Characteristic X-ray energies in eV. Al and Mg Kα are what almost every
#: laboratory instrument uses; the others appear on synchrotron and
#: hard-X-ray machines.
SOURCES: dict[str, float] = {
    "Al": 1486.6,
    "Al Ka": 1486.6,
    "Mg": 1253.6,
    "Mg Ka": 1253.6,
    "Ag La": 2984.3,
    "Cr Ka": 5414.7,
    "Ga Ka": 9251.7,
}

#: Non-monochromatic anodes carry satellite lines: the source itself emits
#: Kα₃ and Kα₄ alongside Kα₁,₂, so every photoelectron peak appears again
#: at LOWER binding energy, displaced by the satellite separation. On a
#: magnesium anode that is +8.4 and +10.1 eV of kinetic energy, which is
#: 8-10 eV *below* the parent peak on a binding-energy plot — right where
#: people look for a reduced species.
XRAY_SATELLITES: dict[str, tuple[tuple[float, float], ...]] = {
    # (displacement in eV above the parent's kinetic energy, relative height)
    "Mg": ((8.4, 0.081), (10.2, 0.041), (17.5, 0.006), (20.0, 0.003)),
    "Al": ((9.8, 0.066), (11.8, 0.032), (20.1, 0.005), (23.4, 0.003)),
}

#: Typical spectrometer work function. Only ever a starting value: it is
#: instrument-specific and drifts, which is why the binding-energy scale is
#: referenced against a known line instead of trusted.
DEFAULT_WORK_FUNCTION = 4.5


class XPSError(ValueError):
    """Raised when a photoelectron spectrum cannot be built or used."""


@dataclass
class XPSSpectrum:
    """One region or survey, on the binding-energy scale."""

    binding_energy: np.ndarray
    """eV. Stored ascending whatever the file's order; plotting reverses it,
    which is a drawing convention and not a property of the data."""
    counts: np.ndarray
    photon_energy: Optional[float] = None
    """eV. ``None`` when the file does not say, which blocks anything that
    needs the kinetic scale rather than guessing aluminium."""
    pass_energy: Optional[float] = None
    """eV. Sets the analyser resolution, and therefore the floor below which
    no fitted FWHM is believable."""
    dwell_s: Optional[float] = None
    sweeps: Optional[int] = None
    work_function: float = DEFAULT_WORK_FUNCTION
    monochromated: bool = True
    """Whether the source is monochromated. A non-monochromated anode puts
    X-ray satellites in the spectrum and they are not chemistry."""
    intensity_unit: str = "cuentas"
    """``"cuentas"`` or ``"cuentas/s"``. Not cosmetic: counting statistics
    only apply to accumulated counts, so a spectrum stored in counts per
    second cannot be judged against √N until it is multiplied back by the
    dwell time and the number of sweeps."""
    region: str = ""
    """What the instrument called it: ``"C1s"``, ``"Survey"``…"""
    name: str = "xps"
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.binding_energy = np.asarray(self.binding_energy, dtype=float).ravel()
        self.counts = np.asarray(self.counts, dtype=float).ravel()
        if self.binding_energy.size != self.counts.size:
            raise XPSError(
                f"el eje tiene {self.binding_energy.size} puntos y las cuentas "
                f"{self.counts.size}"
            )
        if self.binding_energy.size < 5:
            raise XPSError("un espectro necesita al menos cinco puntos")
        if not np.all(np.isfinite(self.binding_energy)):
            raise XPSError("el eje de energía de enlace contiene NaN o inf")
        bad = int(np.count_nonzero(~np.isfinite(self.counts)))
        if bad:
            first = int(np.argmax(~np.isfinite(self.counts)))
            raise XPSError(
                f"{bad} de {self.counts.size} cuentas no son finitas (la "
                f"primera en {self.binding_energy[first]:.1f} eV). Un NaN "
                "atraviesa el fondo y el ajuste y sale como un estado químico "
                "con aspecto de válido"
            )
        if self.photon_energy is not None and self.photon_energy <= 0:
            raise XPSError("la energía del fotón tiene que ser positiva")
        self._sort()

    def _sort(self) -> None:
        """Ascending in binding energy, averaging repeated abscissae."""
        order = np.argsort(self.binding_energy, kind="stable")
        energy = self.binding_energy[order]
        counts = self.counts[order]
        same = np.isclose(np.diff(energy), 0.0, atol=1e-9)
        if same.any():
            unique, index = np.unique(np.round(energy, 9), return_inverse=True)
            summed = np.zeros(unique.size)
            times = np.zeros(unique.size)
            np.add.at(summed, index, counts)
            np.add.at(times, index, 1.0)
            energy, counts = unique, summed / times
        self.binding_energy = energy
        self.counts = counts

    # -- scales --------------------------------------------------------
    @property
    def kinetic_energy(self) -> np.ndarray:
        """``hν − E_B − φ``. Raises if the photon energy is unknown."""
        if self.photon_energy is None:
            raise XPSError(
                f"{self.name}: el archivo no dice la energía del fotón, así "
                "que no hay escala cinética. Pásala con photon_energy= o "
                "declara el ánodo; suponer aluminio mueve cada línea Auger "
                "233 eV si en realidad era magnesio"
            )
        return self.photon_energy - self.binding_energy - self.work_function

    def binding_of_kinetic(self, kinetic: float | np.ndarray) -> np.ndarray:
        """Where a fixed-kinetic-energy feature — an Auger line — appears."""
        if self.photon_energy is None:
            raise XPSError(
                f"{self.name}: sin energía del fotón no se puede situar una "
                "línea Auger, porque su energía de enlace aparente depende de "
                "ella"
            )
        return self.photon_energy - np.asarray(kinetic, dtype=float) - self.work_function

    # -- shape ---------------------------------------------------------
    @property
    def range(self) -> tuple[float, float]:
        return (float(self.binding_energy[0]), float(self.binding_energy[-1]))

    @property
    def step(self) -> float:
        """Median spacing in eV. The resolution floor of everything fitted."""
        return float(np.median(np.diff(self.binding_energy)))

    @property
    def is_survey(self) -> bool:
        """A wide scan. Decided by the span, not by what the file called it.

        Instruments name regions inconsistently and a file called
        ``Survey`` is sometimes a 50 eV window somebody renamed.
        """
        low, high = self.range
        return (high - low) > 200.0

    def covers(self, low: float, high: float, fraction: float = 0.9) -> bool:
        """Whether the measured window really contains a range.

        The same rule the Raman side uses, for the same reason: the absence
        of a peak is only evidence if the region was measured.
        """
        first, last = self.range
        overlap = min(high, last) - max(low, first)
        return overlap > 0 and overlap >= fraction * (high - low)

    def region_of(self, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
        """The points inside a binding-energy window."""
        inside = (self.binding_energy >= min(low, high)) & \
                 (self.binding_energy <= max(low, high))
        return self.binding_energy[inside], self.counts[inside]

    def crop(self, low: float, high: float) -> "XPSSpectrum":
        energy, counts = self.region_of(low, high)
        if energy.size < 5:
            raise XPSError(
                f"la ventana {low:g}–{high:g} eV deja {energy.size} puntos")
        return self.with_counts(counts, f"crop({low:g}, {high:g})", energy)

    def with_counts(self, counts, step: str,
                    energy: Optional[np.ndarray] = None) -> "XPSSpectrum":
        """The same measurement with new values, recording what was done."""
        return XPSSpectrum(
            binding_energy=self.binding_energy if energy is None else energy,
            counts=counts, photon_energy=self.photon_energy,
            pass_energy=self.pass_energy, dwell_s=self.dwell_s,
            sweeps=self.sweeps, work_function=self.work_function,
            monochromated=self.monochromated, intensity_unit=self.intensity_unit,
            region=self.region,
            name=self.name, metadata=dict(self.metadata),
            history=list(self.history) + [step],
        )

    def shifted(self, offset: float, why: str = "") -> "XPSSpectrum":
        """The axis moved by ``offset`` eV — charge referencing.

        Positive moves peaks to higher binding energy. The shift goes in
        the history because a binding energy quoted without saying what it
        was referenced to is not a measurement.
        """
        moved = self.with_counts(self.counts, f"shift({offset:+.3f} eV)"
                                 + (f": {why}" if why else ""))
        moved.binding_energy = self.binding_energy + offset
        moved.metadata["charge_shift_ev"] = (
            float(self.metadata.get("charge_shift_ev", 0.0)) + float(offset))
        if why:
            moved.metadata["charge_reference"] = why
        return moved

    # -- noise ---------------------------------------------------------
    @property
    def accumulated_counts(self) -> Optional[np.ndarray]:
        """The counts actually accumulated, or ``None`` if unknowable.

        A spectrum saved in counts per second has had its counting
        statistics divided away. Multiplying back by dwell time × sweeps
        recovers them; without both numbers the Poisson expectation cannot
        be formed at all, and this returns ``None`` rather than pretending
        the stored values are counts.
        """
        if self.intensity_unit.startswith("cuentas/"):
            if self.dwell_s is None:
                return None
            scale = float(self.dwell_s) * float(self.sweeps or 1)
            if scale <= 0:
                return None
            return self.counts * scale
        return self.counts

    def noise_estimate(self) -> float:
        """Counting noise, from the median absolute second difference.

        Robust to peaks and to any smooth background. For raw counts the
        Poisson expectation is √N, and a measured noise well below that
        means the file has already been smoothed — which inflates every
        uncertainty that comes out of a fit.
        """
        if self.counts.size < 5:
            return 0.0
        second = np.diff(self.counts, n=2)
        return float(np.median(np.abs(second))) / (0.6745 * np.sqrt(6.0))

    def looks_smoothed(self) -> Optional[str]:
        """Whether the counts are quieter than counting statistics allow.

        Returns ``None`` — no verdict — when the spectrum is in counts per
        second and the dwell time is unknown, because then there is no
        Poisson expectation to compare against.
        """
        accumulated = self.accumulated_counts
        if accumulated is None:
            return None
        level = float(np.median(accumulated))
        if level <= 0:
            return None
        second = np.diff(accumulated, n=2)
        measured = float(np.median(np.abs(second))) / (0.6745 * np.sqrt(6.0))
        expected = np.sqrt(level)
        if measured < 0.5 * expected:
            return (
                f"el ruido medido ({measured:.1f}) es menos de la mitad del "
                f"que da la estadística de conteo para {level:.0f} cuentas "
                f"({expected:.1f}): el espectro viene suavizado. Las "
                "incertidumbres de cualquier ajuste saldrán optimistas, "
                "porque el ajuste creerá que hay más información de la que hay"
            )
        return None

    def describe(self) -> str:
        low, high = self.range
        text = (f"{self.name}: {self.counts.size} puntos, {low:.1f}–{high:.1f} eV"
                f", paso {self.step:.3f} eV")
        if self.photon_energy:
            text += f", hν = {self.photon_energy:.1f} eV"
        if self.pass_energy:
            text += f", E_paso {self.pass_energy:.0f} eV"
        if not self.monochromated:
            text += ", fuente no monocromada"
        return text

    def __repr__(self) -> str:            # pragma: no cover - debugging aid
        return f"<XPSSpectrum {self.describe()}>"


def source_energy(name: str) -> float:
    """Photon energy of a named anode."""
    key = name.strip()
    for candidate in (key, key.replace("α", "a").replace("Kα", "Ka")):
        if candidate in SOURCES:
            return SOURCES[candidate]
    short = key.split()[0] if key else ""
    if short in SOURCES:
        return SOURCES[short]
    raise XPSError(
        f"ánodo desconocido: {name!r}; hay {', '.join(sorted(set(SOURCES)))}")


__all__ = [
    "DEFAULT_WORK_FUNCTION",
    "SOURCES",
    "XRAY_SATELLITES",
    "XPSError",
    "XPSSpectrum",
    "source_energy",
]
