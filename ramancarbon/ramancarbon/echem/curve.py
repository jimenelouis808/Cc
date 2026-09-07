"""The measured curves, and the electrode they were measured on.

Three data objects, each validated on construction the way
:class:`ramancarbon.core.spectrum.Spectrum` is, plus an
:class:`Electrode` that carries the things without which most results
cannot be normalised: the active mass, the geometric area, the reference
electrode and the pH.

The :class:`Electrode` is where a lot of quiet errors live, so it is
explicit about all of them:

* **Mass.** A specific capacitance needs the mass of *active material*,
  not of the electrode. Binder and conductive additive are typically 20 %
  of it, and quoting against total mass understates the result by that
  much — which is the one direction of error nobody complains about, and
  therefore the one that spreads.
* **Reference electrode.** A potential quoted against "Ag/AgCl" without
  saying what the filling solution was carries 13 mV of ambiguity, and
  "SCE" is 30 mV away from Ag/AgCl 3 M. Conversion to RHE needs the pH.
* **Uncompensated resistance.** Without iR correction, a Tafel slope is
  wrong by an amount that grows with current, so a catalyst measured
  without it always looks worse at high current than it is. The field
  reports the correction level (85 %, 100 %) as a matter of course, and
  so does this.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..database.loader import DATA_DIR


class CurveError(ValueError):
    """Raised when a measured curve is malformed."""


@lru_cache(maxsize=2)
def load_echem_database(directory: Optional[str] = None) -> dict:
    """The electrochemistry constants file."""
    path = Path(directory or DATA_DIR) / "echem.json"
    if not path.is_file():
        raise CurveError(f"falta el archivo de electroquímica: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise CurveError(f"{name} tiene que ser un vector")
    bad = np.flatnonzero(~np.isfinite(array))
    if bad.size:
        raise CurveError(
            f"{name} contiene valores no finitos; el primero está en el punto "
            f"{int(bad[0])}"
        )
    return array


@dataclass
class Electrode:
    """What the current has to be normalised by, and against what reference."""

    mass_mg: Optional[float] = None
    """Mass of **active material** in mg. Not of the electrode: binder and
    conductive carbon are typically 20 % of that."""
    area_cm2: Optional[float] = None
    """Geometric area in cm². The one benchmark current densities use."""
    volume_cm3: Optional[float] = None
    reference: str = "Ag/AgCl_3M"
    """Key from ``echem.json``. ``"RHE"`` means the data are already there."""
    ph: Optional[float] = None
    """Needed to convert to RHE. Without it the conversion is refused."""
    resistance_ohm: Optional[float] = None
    """Uncompensated series resistance, for iR correction. Take it from the
    high-frequency intercept of an impedance spectrum on the same cell."""
    ir_compensated_fraction: float = 0.0
    """How much the potentiostat already compensated, 0–1."""
    label: str = ""

    def specific(self, value: float, basis: str = "mass") -> Optional[float]:
        """Normalise a quantity, or ``None`` if that basis is unknown."""
        if basis == "mass":
            return None if not self.mass_mg else value / (self.mass_mg * 1e-3)
        if basis == "area":
            return None if not self.area_cm2 else value / self.area_cm2
        if basis == "volume":
            return None if not self.volume_cm3 else value / self.volume_cm3
        raise CurveError(f"base de normalización desconocida: {basis!r}")

    def to_rhe(self, potential: np.ndarray) -> tuple[Optional[np.ndarray], str]:
        """Convert a measured potential to the RHE scale.

        ``E(RHE) = E(medido) + E°(referencia vs SHE) + 0.05916 · pH``.

        Returns ``(converted, explanation)`` and ``(None, reason)`` when the
        conversion cannot be made — which is a refusal on purpose. An
        overpotential is quoted against RHE because the equilibrium
        potentials of the hydrogen and oxygen reactions do not move with pH
        on that scale; guessing a pH to get there would put a systematic
        59 mV per pH unit into every number downstream.
        """
        data = load_echem_database()
        if self.reference == "RHE":
            return np.asarray(potential, dtype=float), "los datos ya están en RHE"
        entry = data["reference_electrodes"].get(self.reference)
        if entry is None:
            return None, (
                f"referencia desconocida {self.reference!r}; disponibles: "
                + ", ".join(sorted(data["reference_electrodes"]))
            )
        if entry.get("vs_she") is None:
            return None, f"{entry['label']} no tiene un potencial fijo frente a SHE"
        if self.ph is None:
            return None, (
                "hace falta el pH para pasar a RHE. Sin él la conversión "
                "arrastra 59 mV por unidad de pH a todo lo que venga después, "
                "así que el programa prefiere no darla"
            )
        slope = float(data["nernst_slope_v_per_ph"])
        offset = float(entry["vs_she"]) + slope * float(self.ph)
        return (
            np.asarray(potential, dtype=float) + offset,
            f"E(RHE) = E({entry['label']}) + {entry['vs_she']:.3f} + "
            f"{slope:.5f}·{self.ph:g} = E + {offset:.4f} V",
        )

    def ir_correct(
        self, potential: np.ndarray, current: np.ndarray
    ) -> tuple[np.ndarray, str]:
        """Remove the ohmic drop that the potentiostat did not.

        ``E_corregido = E − (1 − f) · I · R_u``. Returns the potential and a
        sentence saying what was done, including when nothing was.
        """
        if not self.resistance_ohm:
            return np.asarray(potential, dtype=float), (
                "SIN corrección de caída óhmica: no se ha dado la resistencia "
                "no compensada. La pendiente de Tafel sale entonces demasiado "
                "grande, y el error crece con la corriente — así que el "
                "catalizador parece peor de lo que es justo donde más importa"
            )
        remaining = 1.0 - float(self.ir_compensated_fraction)
        corrected = np.asarray(potential, dtype=float) - remaining * np.asarray(
            current, dtype=float
        ) * float(self.resistance_ohm)
        return corrected, (
            f"corregido el {100 * remaining:.0f} % restante de la caída óhmica "
            f"con R_u = {self.resistance_ohm:g} Ω"
        )

    def describe(self) -> str:
        parts = []
        if self.mass_mg:
            parts.append(f"masa activa {self.mass_mg:g} mg")
        if self.area_cm2:
            parts.append(f"área {self.area_cm2:g} cm²")
        if self.volume_cm3:
            parts.append(f"volumen {self.volume_cm3:g} cm³")
        parts.append(f"referencia {self.reference}")
        if self.ph is not None:
            parts.append(f"pH {self.ph:g}")
        if self.resistance_ohm:
            parts.append(
                f"R_u {self.resistance_ohm:g} Ω "
                f"({100 * self.ir_compensated_fraction:.0f} % compensada)"
            )
        return "Electrodo: " + ", ".join(parts) if parts else "Electrodo: sin datos"


@dataclass
class Voltammogram:
    """A cyclic voltammogram: current against potential at a fixed sweep rate."""

    potential: np.ndarray
    """V, on whatever reference the electrode declares."""
    current: np.ndarray
    """A. Positive is anodic (oxidation), the IUPAC convention."""
    scan_rate: float
    """V/s. Not mV/s — the unit mismatch is a factor of 1000 and the most
    common single error in this whole analysis, so it is checked."""
    electrode: Electrode = field(default_factory=Electrode)
    cycle: Optional[np.ndarray] = None
    name: str = "CV"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.potential = _finite("el potencial", self.potential)
        self.current = _finite("la corriente", self.current)
        if self.potential.size != self.current.size:
            raise CurveError(
                f"potencial ({self.potential.size} puntos) y corriente "
                f"({self.current.size}) no coinciden"
            )
        if self.potential.size < 10:
            raise CurveError("un voltamperograma de menos de 10 puntos no sirve")
        if self.scan_rate <= 0.0:
            raise CurveError("la velocidad de barrido tiene que ser positiva")
        if self.scan_rate > 20.0:
            raise CurveError(
                f"velocidad de barrido de {self.scan_rate} V/s. Si querías decir "
                f"{self.scan_rate:g} mV/s, son {self.scan_rate / 1000:g} V/s: "
                "esta clase trabaja en V/s. Es el error de unidades más "
                "frecuente de todo el análisis, y una capacitancia calculada "
                "con la velocidad en mV/s sale 1000 veces demasiado pequeña"
            )
        if self.scan_rate > 1.0:
            self.metadata.setdefault(
                "aviso_velocidad",
                f"{self.scan_rate:g} V/s es una velocidad muy alta. Es "
                "legítima en voltamperometría rápida, pero si venía de una "
                "hoja en mV/s, aquí está el factor de 1000",
            )
        if self.cycle is not None:
            self.cycle = np.asarray(self.cycle)
            if self.cycle.size != self.potential.size:
                raise CurveError("el vector de ciclos no coincide con los datos")

    @property
    def window(self) -> tuple[float, float]:
        return float(self.potential.min()), float(self.potential.max())

    @property
    def span(self) -> float:
        low, high = self.window
        return high - low

    @property
    def n(self) -> int:
        return int(self.potential.size)

    def cycles(self) -> list["Voltammogram"]:
        """Split into individual cycles.

        Uses the recorded cycle column when there is one and detects sweep
        reversals otherwise. The **first cycle is different from the rest**
        — it includes irreversible formation, SEI growth or oxide
        conditioning — so the analysis defaults to the last one and says so
        rather than averaging the two together.
        """
        if self.cycle is not None:
            labels = np.unique(self.cycle)
            return [self._subset(self.cycle == label, f"ciclo {label}")
                    for label in labels]
        direction = np.sign(np.diff(self.potential))
        direction[direction == 0] = 1
        turns = np.flatnonzero(np.diff(direction) != 0) + 1
        if turns.size < 2:
            return [self]
        # Two turning points per cycle.
        bounds = [0, *turns[1::2].tolist(), self.n]
        out: list[Voltammogram] = []
        for index in range(len(bounds) - 1):
            mask = np.zeros(self.n, dtype=bool)
            mask[bounds[index]:bounds[index + 1]] = True
            if mask.sum() >= 10:
                out.append(self._subset(mask, f"ciclo {index + 1}"))
        return out or [self]

    def sweeps(self) -> tuple["Voltammogram", "Voltammogram"]:
        """``(anodic, cathodic)``: the forward and reverse halves."""
        direction = np.sign(np.gradient(self.potential))
        forward = direction > 0
        backward = ~forward
        if forward.sum() < 5 or backward.sum() < 5:
            raise CurveError(
                "no se distinguen las dos ramas del barrido; ¿es un "
                "voltamperograma cíclico completo?"
            )
        return self._subset(forward, "anódica"), self._subset(backward, "catódica")

    def _subset(self, mask: np.ndarray, suffix: str) -> "Voltammogram":
        return Voltammogram(
            potential=self.potential[mask],
            current=self.current[mask],
            scan_rate=self.scan_rate,
            electrode=self.electrode,
            cycle=None if self.cycle is None else self.cycle[mask],
            name=f"{self.name} [{suffix}]",
            metadata=dict(self.metadata),
        )

    def describe(self) -> str:
        low, high = self.window
        return (
            f"{self.name}: {self.n} puntos, {low:+.3f} a {high:+.3f} V "
            f"(ventana {self.span:.3f} V), {1000 * self.scan_rate:g} mV/s, "
            f"|I|max {1000 * np.abs(self.current).max():.3g} mA"
        )


@dataclass
class ChargeDischarge:
    """A galvanostatic charge–discharge curve."""

    time: np.ndarray
    """s, monotonically increasing."""
    potential: np.ndarray
    current: np.ndarray
    """A, signed: positive charging, negative discharging."""
    electrode: Electrode = field(default_factory=Electrode)
    cycle: Optional[np.ndarray] = None
    name: str = "GCD"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = _finite("el tiempo", self.time)
        self.potential = _finite("el potencial", self.potential)
        self.current = _finite("la corriente", self.current)
        sizes = {self.time.size, self.potential.size, self.current.size}
        if len(sizes) != 1:
            raise CurveError("tiempo, potencial y corriente no tienen igual longitud")
        if self.time.size < 10:
            raise CurveError("una curva de menos de 10 puntos no sirve")
        if np.any(np.diff(self.time) < 0):
            raise CurveError("el tiempo no es monótono creciente")
        if self.cycle is not None:
            self.cycle = np.asarray(self.cycle)

    @property
    def n(self) -> int:
        return int(self.time.size)

    def segments(self) -> list[tuple[str, "ChargeDischarge"]]:
        """Split into charge and discharge branches by the sign of the current.

        Segments shorter than five points are dropped: they are the
        switching transients, and a "discharge" three points long produces
        a capacitance of whatever you like.
        """
        sign = np.sign(self.current)
        sign[sign == 0] = 1
        breaks = np.flatnonzero(np.diff(sign) != 0) + 1
        bounds = [0, *breaks.tolist(), self.n]
        out: list[tuple[str, ChargeDischarge]] = []
        for index in range(len(bounds) - 1):
            start, stop = bounds[index], bounds[index + 1]
            if stop - start < 5:
                continue
            label = "carga" if sign[start] > 0 else "descarga"
            out.append((label, self._subset(slice(start, stop), f"{label} {index + 1}")))
        return out

    def _subset(self, where, suffix: str) -> "ChargeDischarge":
        return ChargeDischarge(
            time=self.time[where],
            potential=self.potential[where],
            current=self.current[where],
            electrode=self.electrode,
            cycle=None if self.cycle is None else self.cycle[where],
            name=f"{self.name} [{suffix}]",
            metadata=dict(self.metadata),
        )

    def describe(self) -> str:
        return (
            f"{self.name}: {self.n} puntos, {self.time[-1] - self.time[0]:.1f} s, "
            f"{self.potential.min():+.3f} a {self.potential.max():+.3f} V, "
            f"|I| {1000 * np.abs(self.current).mean():.3g} mA"
        )


@dataclass
class Impedance:
    """An electrochemical impedance spectrum."""

    frequency: np.ndarray
    """Hz, any order; sorted descending on construction, as measured."""
    z: np.ndarray
    """Complex impedance in Ω, with the physical sign: a capacitive
    response has **negative** imaginary part. Nyquist plots show −Z″, which
    is a plotting convention and not a change of the data — storing the
    negated value is a standing source of sign errors in circuit fits."""
    electrode: Electrode = field(default_factory=Electrode)
    name: str = "EIS"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frequency = _finite("la frecuencia", self.frequency)
        z = np.asarray(self.z, dtype=complex)
        if z.size != self.frequency.size:
            raise CurveError("frecuencia e impedancia no tienen igual longitud")
        if not np.all(np.isfinite(z.real)) or not np.all(np.isfinite(z.imag)):
            raise CurveError("la impedancia contiene valores no finitos")
        if np.any(self.frequency <= 0.0):
            raise CurveError("hay frecuencias nulas o negativas")
        if self.frequency.size < 5:
            raise CurveError("un espectro de menos de 5 frecuencias no sirve")
        order = np.argsort(-self.frequency)
        self.frequency = self.frequency[order]
        self.z = z[order]

    @property
    def omega(self) -> np.ndarray:
        return 2.0 * math.pi * self.frequency

    @property
    def n(self) -> int:
        return int(self.frequency.size)

    @property
    def modulus(self) -> np.ndarray:
        return np.abs(self.z)

    @property
    def phase_deg(self) -> np.ndarray:
        return np.degrees(np.angle(self.z))

    def describe(self) -> str:
        return (
            f"{self.name}: {self.n} frecuencias, "
            f"{self.frequency.min():.3g}–{self.frequency.max():.3g} Hz, "
            f"|Z| {self.modulus.min():.3g}–{self.modulus.max():.3g} Ω"
        )


__all__ = [
    "ChargeDischarge",
    "CurveError",
    "Electrode",
    "Impedance",
    "Voltammogram",
    "load_echem_database",
]
