"""Diffusion coefficients, differential capacity and rate capability.

The measurements in this module all answer the same question — *how fast
can charge get in and out, and what limits it* — from four different
experiments. They share one hazard: every one of them ends in a diffusion
coefficient, and a diffusion coefficient is the quantity in
electrochemistry most often quoted to three figures and wrong by three
orders of magnitude, because it depends on an **area** and an **amount**
that were guessed.

The area is the worst of it. A porous electrode's electrochemically
active area is ten to a thousand times its geometric one, and
Randles–Ševčík divides by the area squared. Nothing here computes a
diffusion coefficient without being told which area it is using, and
every result says so.

What is here:

``randles_sevcik``
    D from how the peak current grows with the square root of scan rate.
``warburg_diffusion``
    D from the Warburg coefficient of an impedance spectrum, which needs
    the same area and, additionally, the slope of the potential against
    composition.
``gitt``
    D from a current pulse and the relaxation after it. The most direct of
    the three, and the one whose assumptions are easiest to check.
``differential_capacity``
    dQ/dV and dV/dQ, which turn a featureless charge–discharge curve into
    a set of peaks that can be assigned to phase transitions.
``rate_capability`` and ``ragone``
    What is left at high rate, and the energy–power trade-off.
``fade_model``
    Fitting capacity loss over cycles, which distinguishes a mechanism
    that ends from one that does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .curve import ChargeDischarge, Impedance

#: Faraday constant, C/mol.
FARADAY = 96485.332
#: Gas constant, J/(mol K).
GAS_CONSTANT = 8.314462618
#: The Randles–Ševčík constant at 25 °C, for a reversible system:
#: ``i_p = 2.69e5 · n^1.5 · A · D^0.5 · C · v^0.5`` in A, cm², cm²/s,
#: mol/cm³ and V/s.
RANDLES_SEVCIK_25C = 2.69e5


@dataclass
class Diffusion:
    """A diffusion coefficient and everything it depends on."""

    d_cm2_s: Optional[float]
    method: str
    area_cm2: float = 0.0
    area_basis: str = "geométrica"
    """Which area was used. The single largest source of error in every
    one of these methods, and never inferred."""
    r_squared: Optional[float] = None
    available: bool = True
    reason: str = ""
    detail: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.available or self.d_cm2_s is None:
            return f"{self.method}: no disponible ({self.reason})"
        text = f"{self.method}: D = {self.d_cm2_s:.3g} cm²/s"
        if self.r_squared is not None:
            text += f" (R² = {self.r_squared:.4f})"
        text += f", área {self.area_basis} {self.area_cm2:g} cm²"
        return text


def _area_warning(area_basis: str) -> list[str]:
    if area_basis.startswith("geom"):
        return [
            "el área usada es la GEOMÉTRICA: en un electrodo poroso la "
            "superficie electroquímicamente activa es de diez a mil veces "
            "mayor, y estos métodos dividen por el área al cuadrado, así que "
            "el D que sale puede estar seis órdenes de magnitud alto. Es un "
            "número para comparar TUS muestras entre sí, no para publicar "
            "como propiedad del material"
        ]
    return [
        f"el área usada es la {area_basis}: dila junto al D, porque el mismo "
        "experimento con el área geométrica da un número muy distinto"
    ]


def randles_sevcik(
    scan_rates: Sequence[float],
    peak_currents: Sequence[float],
    concentration_mol_cm3: float,
    area_cm2: float,
    electrons: int = 1,
    area_basis: str = "geométrica",
    reversible: bool = True,
) -> Diffusion:
    """D from the slope of peak current against the square root of scan rate.

    ``i_p = 2.69·10⁵ n^{3/2} A D^{1/2} C v^{1/2}`` for a reversible
    system. The **linearity is the test of the method**, not a detail: if
    the peak current is not proportional to √v the process is not
    diffusion-limited, and then the number this returns is not a diffusion
    coefficient at all — it is the slope of a line through points that do
    not lie on one.

    ``reversible=False`` uses the irreversible form, whose constant is
    2.99·10⁵·α^{1/2} — and since α is rarely known better than a factor of
    two, the result is quoted with that stated.
    """
    rates = np.asarray(scan_rates, dtype=float)
    currents = np.abs(np.asarray(peak_currents, dtype=float))
    if rates.size != currents.size:
        return Diffusion(None, "Randles-Ševčík", available=False,
                         reason=f"hay {rates.size} velocidades y "
                                f"{currents.size} corrientes")
    if rates.size < 3:
        return Diffusion(None, "Randles-Ševčík", available=False,
                         reason="hacen falta al menos tres velocidades")
    if concentration_mol_cm3 <= 0 or area_cm2 <= 0:
        return Diffusion(None, "Randles-Ševčík", available=False,
                         reason="la concentración y el área tienen que ser "
                                "positivas")

    root = np.sqrt(rates)
    slope, intercept = np.polyfit(root, currents, 1)
    predicted = slope * root + intercept
    total = float(np.sum((currents - np.mean(currents)) ** 2))
    r_squared = (1.0 - float(np.sum((currents - predicted) ** 2)) / total
                 if total > 0 else 0.0)

    constant = (RANDLES_SEVCIK_25C * electrons ** 1.5 if reversible
                else 2.99e5 * math.sqrt(0.5) * electrons ** 1.5)
    denominator = constant * area_cm2 * concentration_mol_cm3
    if slope <= 0 or denominator <= 0:
        return Diffusion(None, "Randles-Ševčík", area_cm2=area_cm2,
                         area_basis=area_basis, available=False,
                         r_squared=r_squared,
                         reason="la corriente de pico no crece con √v: el "
                                "proceso no está limitado por difusión y este "
                                "método no se le aplica")

    diffusion = (slope / denominator) ** 2
    warnings = _area_warning(area_basis)
    if r_squared < 0.98:
        warnings.append(
            f"R² = {r_squared:.3f} en i_p frente a √v: la proporcionalidad es "
            "la comprobación de que el proceso está limitado por difusión, y "
            "no se cumple. El número de abajo es la pendiente de una recta "
            "por puntos que no están en una recta"
        )
    if abs(intercept) > 0.2 * float(np.max(currents)):
        warnings.append(
            "la recta no pasa por el origen: hay una contribución capacitiva "
            "grande superpuesta al pico, y hay que restarla antes de que la "
            "pendiente signifique algo"
        )
    if not reversible:
        warnings.append(
            "forma irreversible: lleva dentro α = 0.5. Como α rara vez se "
            "conoce mejor que un factor de dos, y entra bajo raíz, el D "
            "arrastra ese factor"
        )
    return Diffusion(
        d_cm2_s=diffusion, method="Randles-Ševčík", area_cm2=area_cm2,
        area_basis=area_basis, r_squared=r_squared, warnings=warnings,
        detail={"pendiente_A_por_raizVs": float(slope),
                "corte_A": float(intercept)},
    )


def warburg_coefficient(
    spectrum: Impedance,
    low_frequency_hz: float = 1.0,
    high_frequency_hz: float = 0.1,
) -> tuple[Optional[float], float, list[str]]:
    """σ from the slope of Z′ and −Z″ against ω^(−1/2).

    In the Warburg region both parts of the impedance fall as ω^(−1/2)
    with the *same* coefficient. Fitting them separately and comparing is
    the check that the region chosen really is the Warburg one: if the two
    slopes disagree, what was fitted is the tail of the charge-transfer
    semicircle.
    """
    frequency = np.asarray(spectrum.frequency, dtype=float)
    inside = (frequency <= max(low_frequency_hz, high_frequency_hz)) & \
             (frequency >= min(low_frequency_hz, high_frequency_hz))
    if np.count_nonzero(inside) < 4:
        return None, 0.0, [
            f"solo hay {int(np.count_nonzero(inside))} puntos entre "
            f"{high_frequency_hz:g} y {low_frequency_hz:g} Hz; hacen falta al "
            "menos cuatro para una pendiente"
        ]

    omega = 2.0 * math.pi * frequency[inside]
    basis = 1.0 / np.sqrt(omega)
    real = np.asarray(spectrum.z.real, dtype=float)[inside]
    imaginary = -np.asarray(spectrum.z.imag, dtype=float)[inside]

    slope_real = float(np.polyfit(basis, real, 1)[0])
    slope_imag = float(np.polyfit(basis, imaginary, 1)[0])
    warnings: list[str] = []
    if slope_real <= 0 or slope_imag <= 0:
        return None, 0.0, [
            "las pendientes frente a ω^(−1/2) no son positivas: la ventana "
            "elegida no está en la región de Warburg"
        ]
    ratio = slope_real / slope_imag
    if not 0.7 <= ratio <= 1.4:
        warnings.append(
            f"las pendientes de Z′ y −Z″ difieren un {100 * abs(ratio - 1):.0f} %: "
            "en la región de Warburg tienen que coincidir, así que lo que se "
            "ha ajustado es la cola del semicírculo de transferencia de carga, "
            "no difusión"
        )
    return 0.5 * (slope_real + slope_imag), ratio, warnings


def warburg_diffusion(
    spectrum: Impedance,
    concentration_mol_cm3: float,
    area_cm2: float,
    electrons: int = 1,
    temperature_k: float = 298.15,
    low_frequency_hz: float = 1.0,
    high_frequency_hz: float = 0.1,
    area_basis: str = "geométrica",
) -> Diffusion:
    """D from the Warburg coefficient.

    ``D = (RT / (√2 n²F²A C σ))²``. Assumes a semi-infinite planar
    diffusion field, which a porous electrode is not — one more reason the
    number is for comparing samples rather than for publishing as a
    property.
    """
    sigma, ratio, warnings = warburg_coefficient(
        spectrum, low_frequency_hz, high_frequency_hz)
    if sigma is None:
        return Diffusion(None, "Warburg", area_cm2=area_cm2,
                         area_basis=area_basis, available=False,
                         reason=warnings[0] if warnings else "sin pendiente")
    if concentration_mol_cm3 <= 0 or area_cm2 <= 0:
        return Diffusion(None, "Warburg", available=False,
                         reason="la concentración y el área tienen que ser "
                                "positivas")

    numerator = GAS_CONSTANT * temperature_k
    denominator = (math.sqrt(2.0) * (electrons * FARADAY) ** 2
                   * area_cm2 * concentration_mol_cm3 * sigma)
    diffusion = (numerator / denominator) ** 2
    warnings = warnings + _area_warning(area_basis) + [
        "supone difusión semiinfinita en un plano; un electrodo poroso no lo "
        "es, y el D sale bajo por la tortuosidad del poro"
    ]
    return Diffusion(
        d_cm2_s=diffusion, method="Warburg", area_cm2=area_cm2,
        area_basis=area_basis, warnings=warnings,
        detail={"sigma_ohm_s_medio": float(sigma), "cociente_pendientes": ratio},
    )


def _unfinished_relaxation(time: np.ndarray, potential: np.ndarray) -> float:
    """How much further the potential would still move, in volts.

    An exponential is fitted to the rest and its asymptote compared with
    the last measured point. Fitted on the LOG of the remaining change,
    which needs no starting guess.
    """
    if time.size < 6:
        return 0.0
    span = float(potential[-1] - potential[0])
    if abs(span) < 1e-9:
        return 0.0
    # Guess the asymptote by extrapolating the last third's own decay.
    third = potential[-max(3, potential.size // 3):]
    moments = time[-third.size:]
    slope = float(np.polyfit(moments, third, 1)[0])
    if abs(slope) < 1e-12:
        return 0.0
    # Local time constant from the curvature of the last third.
    difference = np.gradient(third, moments)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.gradient(difference, moments) / np.where(
            np.abs(difference) > 1e-15, difference, np.nan)
    finite = ratio[np.isfinite(ratio) & (ratio < 0)]
    if finite.size == 0:
        return 0.0
    tau = -1.0 / float(np.median(finite))
    return abs(slope * tau)


@dataclass
class GITTStep:
    """One pulse-and-rest of a GITT experiment."""

    d_cm2_s: Optional[float]
    potential_v: float
    delta_es: float
    """Steady-state potential change over the step."""
    delta_et: float
    """Potential change during the pulse itself, IR drop excluded."""
    tau_s: float = 0.0
    linear_r2: Optional[float] = None
    warnings: list[str] = field(default_factory=list)


def gitt(
    time_s: Sequence[float],
    potential_v: Sequence[float],
    current_a: float,
    pulse_s: float,
    molar_volume_cm3: float,
    mass_g: float,
    molar_mass_g: float,
    area_cm2: float,
    area_basis: str = "geométrica",
) -> Diffusion:
    """D from one galvanostatic pulse and the relaxation after it.

    ``D = (4/πτ)(n_m V_m / A)² (ΔE_s / ΔE_τ)²`` — Weppner and Huggins.
    Two conditions have to hold and both are checked rather than assumed:

    - the pulse must be **short compared with the diffusion time**, so
      that the potential during it varies as √t. The linearity of E
      against √t over the pulse is the test, and it is reported;
    - the relaxation must reach a **steady state**, or ΔE_s is not the
      thermodynamic change but wherever the measurement was stopped.
    """
    time = np.asarray(time_s, dtype=float)
    potential = np.asarray(potential_v, dtype=float)
    if time.size != potential.size or time.size < 10:
        return Diffusion(None, "GITT", available=False,
                         reason="hacen falta al menos diez puntos de tiempo y "
                                "potencial de la misma longitud")
    if pulse_s <= 0 or molar_volume_cm3 <= 0 or area_cm2 <= 0 or mass_g <= 0:
        return Diffusion(None, "GITT", available=False,
                         reason="pulso, volumen molar, masa y área tienen que "
                                "ser positivos")

    start = float(time[0])
    during = (time - start) <= pulse_s
    after = ~during
    if np.count_nonzero(during) < 4 or np.count_nonzero(after) < 4:
        return Diffusion(None, "GITT", available=False,
                         reason="el pulso o el reposo tienen menos de cuatro "
                                "puntos")

    # The IR drop is the jump in the first instants; excluded by fitting
    # E against sqrt(t) over the pulse and taking the FITTED change, not
    # the raw one. Including it inflates dE_tau and so lowers D by its
    # square.
    root = np.sqrt(time[during] - start + 1e-12)
    slope, intercept = np.polyfit(root, potential[during], 1)
    predicted = slope * root + intercept
    total = float(np.sum((potential[during] - np.mean(potential[during])) ** 2))
    r_squared = (1.0 - float(np.sum((potential[during] - predicted) ** 2)) / total
                 if total > 0 else 0.0)
    delta_et = abs(slope) * math.sqrt(pulse_s)

    rest = potential[after]
    delta_es = abs(float(rest[-1]) - float(potential[0]))
    warnings: list[str] = []

    # Whether the relaxation finished is decided by extrapolating it, not
    # by looking at how much the last few points moved. A rest cut off at
    # a third of its time constant has a nearly flat tail and is nowhere
    # near equilibrium; the tail only looks quiet because the exponential
    # is slow, and reading it as settled is what makes DeltaE_s — and so D
    # — come out wherever the measurement happened to stop.
    remaining = _unfinished_relaxation(time[after] - start, rest)
    if delta_es > 0 and remaining > 0.05 * delta_es:
        warnings.append(
            f"extrapolando el reposo, al potencial le quedan unos "
            f"{1e3 * remaining:.1f} mV por moverse: no ha llegado al "
            "equilibrio, así que ΔE_s no es el cambio termodinámico sino "
            "donde se paró de medir"
        )
    if r_squared < 0.98:
        warnings.append(
            f"E frente a √t durante el pulso da R² = {r_squared:.3f}: la "
            "condición de pulso corto no se cumple, y la fórmula de Weppner-"
            "Huggins la supone"
        )
    if delta_et <= 0:
        return Diffusion(None, "GITT", area_cm2=area_cm2,
                         area_basis=area_basis, available=False,
                         reason="el potencial no cambia durante el pulso")

    moles = mass_g / molar_mass_g
    factor = (4.0 / (math.pi * pulse_s)) * (moles * molar_volume_cm3
                                            / area_cm2) ** 2
    diffusion = factor * (delta_es / delta_et) ** 2
    warnings += _area_warning(area_basis)
    return Diffusion(
        d_cm2_s=diffusion, method="GITT", area_cm2=area_cm2,
        area_basis=area_basis, r_squared=r_squared, warnings=warnings,
        detail={"delta_Es_V": delta_es, "delta_Etau_V": delta_et,
                "pulso_s": float(pulse_s), "corriente_A": float(current_a)},
    )


# -- differential capacity ---------------------------------------------

@dataclass
class DifferentialCapacity:
    """dQ/dV and dV/dQ, with the peaks they were computed to show."""

    potential_v: np.ndarray
    dq_dv: np.ndarray
    capacity_ah: np.ndarray
    dv_dq: np.ndarray
    peaks_v: list[float] = field(default_factory=list)
    smoothing: int = 0
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (f"dQ/dV: {len(self.peaks_v)} picos "
                f"({', '.join(f'{v:.3f} V' for v in self.peaks_v)})"
                if self.peaks_v else "dQ/dV: sin picos claros")


def differential_capacity(
    curve: ChargeDischarge,
    branch: str = "descarga",
    smoothing: Optional[int] = None,
    min_prominence: float = 0.1,
    separation_fraction: float = 0.03,
) -> DifferentialCapacity:
    """dQ/dV and dV/dQ from a charge–discharge curve.

    A galvanostatic curve is nearly featureless to the eye and its
    derivative is not: a plateau becomes a peak, and the peaks can be
    assigned to phase transitions and watched as they fade over cycling.

    **The derivative amplifies the noise, and the smoothing is therefore
    part of the measurement, not a cosmetic step.** Too little and the
    result is a comb of noise peaks; too much and two nearby transitions
    merge into one. The window is chosen from the data's own noise when
    not given, and it is reported either way — a dQ/dV curve published
    without its smoothing cannot be compared with anybody else's.
    """
    time = np.asarray(curve.time, dtype=float)
    potential = np.asarray(curve.potential, dtype=float)
    current = np.asarray(curve.current, dtype=float)
    if current.size == 1:
        current = np.full_like(time, float(current[0]))

    if branch.startswith("desc"):
        mask = current < 0
    elif branch.startswith("carg"):
        mask = current > 0
    else:
        raise ValueError(f"rama desconocida: {branch!r}; usa carga o descarga")
    if np.count_nonzero(mask) < 20:
        raise ValueError(
            f"la rama de {branch} tiene {int(np.count_nonzero(mask))} puntos; "
            "hacen falta al menos veinte para derivar"
        )

    # ONE branch, not every discharge in the file glued together. A
    # multi-cycle file has several, and concatenating them puts a step in
    # the potential at each join: the derivative there is enormous, the
    # automatic smoothing chases it, and what comes out is a comb of
    # seventy-eight "peaks" instead of the two transitions that are there.
    span = _longest_run(mask)
    if span is None:
        raise ValueError(f"no hay ningún tramo continuo de {branch}")
    start, stop = span
    if stop - start < 20:
        raise ValueError(
            f"el tramo continuo de {branch} más largo tiene {stop - start} "
            "puntos; hacen falta al menos veinte para derivar"
        )
    time = time[start:stop]
    potential = potential[start:stop]
    current = current[start:stop]
    charge_ah = np.abs(np.concatenate(
        [[0.0], np.cumsum(np.abs(current[1:]) * np.diff(time))])) / 3600.0

    window = smoothing if smoothing is not None else _auto_window(potential)
    smoothed_v = _moving_average(potential, window)
    smoothed_q = _moving_average(charge_ah, window)

    with np.errstate(divide="ignore", invalid="ignore"):
        dv = np.gradient(smoothed_v)
        dq = np.gradient(smoothed_q)
        floor = 1e-6 * max(float(np.ptp(smoothed_v)), 1e-9)
        dq_dv = np.where(np.abs(dv) > floor, dq / dv, np.nan)
        dv_dq = np.where(np.abs(dq) > 1e-15, dv / dq, np.nan)

    finite = np.isfinite(dq_dv)
    peaks = _peaks_of(np.abs(dq_dv[finite]), smoothed_v[finite],
                      min_prominence, separation_fraction)

    warnings = [
        f"suavizado con una ventana de {window} puntos: una curva dQ/dV sin "
        "decir su suavizado no se puede comparar con la de nadie, porque la "
        "altura de los picos depende de él"
    ]
    if window >= 0.1 * potential.size:
        warnings.append(
            "el suavizado se lleva más de un diez por ciento de la rama: dos "
            "transiciones próximas saldrían fundidas en un pico"
        )
    return DifferentialCapacity(
        potential_v=smoothed_v, dq_dv=dq_dv, capacity_ah=smoothed_q,
        dv_dq=dv_dq, peaks_v=peaks, smoothing=window, warnings=warnings,
    )


def _peaks_of(values: np.ndarray, positions: np.ndarray,
              min_prominence: float, separation_fraction: float = 0.03,
              min_separation: int = 5) -> list[float]:
    """Local maxima that stand above their own surroundings.

    A bare "greater than both neighbours" test on a differentiated curve
    returns one peak per noise excursion. Three conditions are added:

    - the peak must rise above its own *local* background by a fraction of
      the largest peak;
    - the ends are excluded, because a discharge curve flattens as it runs
      out and the last point is then greater than its neighbour;
    - peaks closer together than ``separation_fraction`` of the potential
      window are one peak. A broad transition carries several local
      maxima on its top, and two dQ/dV features fifteen millivolts apart
      are not resolved by a noisy galvanostatic curve anyway.
    """
    if values.size < 10:
        return []
    tallest = float(np.max(values))
    if tallest <= 0:
        return []
    threshold = min_prominence * tallest
    # The ends are excluded. A discharge curve flattens as it runs out, so
    # |dQ/dV| rises monotonically into the last point and the last point
    # is then "greater than its neighbour" — a peak at the edge of the
    # window is the window, not a transition.
    margin = max(2, int(0.03 * values.size))
    found: list[tuple[float, float]] = []
    for index in range(margin, values.size - margin):
        if not (values[index] > values[index - 1]
                and values[index] >= values[index + 1]):
            continue
        low = max(0, index - 4 * min_separation)
        high = min(values.size, index + 4 * min_separation)
        floor = float(np.min(values[low:high]))
        if values[index] - floor < threshold:
            continue
        found.append((float(values[index]), float(positions[index])))
    found.sort(reverse=True)
    minimum = separation_fraction * float(np.ptp(positions))
    kept: list[tuple[float, float]] = []
    for height, position in found:
        if all(abs(position - other) > minimum for _, other in kept):
            kept.append((height, position))
    return [position for _, position in sorted(kept, key=lambda item: item[1])]


def _longest_run(mask: np.ndarray) -> Optional[tuple[int, int]]:
    """The longest contiguous stretch of ``True``, as ``(start, stop)``."""
    best: Optional[tuple[int, int]] = None
    start: Optional[int] = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    if start is not None and (best is None or mask.size - start > best[1] - best[0]):
        best = (start, int(mask.size))
    return best


def _auto_window(values: np.ndarray) -> int:
    """A smoothing window from the data's own noise.

    The noise is the median absolute second difference; the window is
    chosen so that averaging brings it below a thousandth of the range,
    which is about where the derivative stops being dominated by it.
    """
    if values.size < 10:
        return 3
    second = np.diff(values, n=2)
    noise = float(np.median(np.abs(second))) / (0.6745 * math.sqrt(6.0))
    span = float(np.ptp(values))
    if noise <= 0 or span <= 0:
        return 5
    needed = (noise / (1e-3 * span)) ** 2
    window = int(np.clip(round(needed), 3, max(5, values.size // 8)))
    return window if window % 2 else window + 1


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:values.size]


# -- rate capability ----------------------------------------------------

@dataclass
class RateCapability:
    """What is left of the capacity at high rate."""

    rates: np.ndarray
    capacities: np.ndarray
    retention: np.ndarray
    """Fraction of the lowest-rate capacity."""
    recovered: Optional[float] = None
    """Capacity when the rate returns to the first one, as a fraction. The
    number that separates a kinetic limitation from damage."""
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        text = (f"retención {100 * self.retention[-1]:.0f} % al pasar de "
                f"{self.rates[0]:g} a {self.rates[-1]:g}")
        if self.recovered is not None:
            text += f", recupera {100 * self.recovered:.0f} % al volver"
        return text


def rate_capability(
    rates: Sequence[float],
    capacities: Sequence[float],
    returned_capacity: Optional[float] = None,
) -> RateCapability:
    """Capacity against rate, with the recovery that tells you why.

    A material that gives back 95 % of its capacity when the rate returns
    to the starting one was **kinetically limited**; one that gives back
    60 % was **damaged**, and the rate test was a degradation experiment.
    The two look identical on the way up.
    """
    rate_array = np.asarray(rates, dtype=float)
    capacity_array = np.asarray(capacities, dtype=float)
    if rate_array.size != capacity_array.size or rate_array.size < 2:
        raise ValueError("hacen falta al menos dos pares (velocidad, capacidad)")
    reference = float(capacity_array[0])
    if reference <= 0:
        raise ValueError("la capacidad de referencia tiene que ser positiva")

    retention = capacity_array / reference
    recovered = (float(returned_capacity) / reference
                 if returned_capacity is not None else None)
    warnings: list[str] = []
    if recovered is not None:
        if recovered < 0.9:
            warnings.append(
                f"al volver a {rate_array[0]:g} solo se recupera el "
                f"{100 * recovered:.0f} %: la pérdida a velocidad alta no era "
                "solo cinética, el electrodo se ha degradado durante la prueba"
            )
        else:
            warnings.append(
                f"se recupera el {100 * recovered:.0f} % al volver: la caída "
                "era cinética, no daño"
            )
    else:
        warnings.append(
            "sin el ciclo de vuelta a la velocidad inicial no se puede "
            "distinguir una limitación cinética de una degradación, y las dos "
            "dan la misma curva de subida"
        )
    if np.any(np.diff(retention) > 0.02):
        warnings.append(
            "la capacidad sube en algún salto de velocidad: suele ser el "
            "electrodo terminando de mojarse o de activarse, y entonces la "
            "primera medida no es la referencia"
        )
    return RateCapability(rates=rate_array, capacities=capacity_array,
                          retention=retention, recovered=recovered,
                          warnings=warnings)


def ragone(
    energies_wh_kg: Sequence[float],
    powers_w_kg: Sequence[float],
    basis: str = "masa de material activo",
) -> dict[str, object]:
    """Energy against power, and the caveat that makes it comparable.

    A Ragone plot is the standard comparison and the standard trap: the
    same cell plotted per gram of active material and per kilogram of
    packaged cell differs by a factor of three to five, and published
    plots mix the two freely.
    """
    energy = np.asarray(energies_wh_kg, dtype=float)
    power = np.asarray(powers_w_kg, dtype=float)
    if energy.size != power.size or energy.size < 2:
        raise ValueError("hacen falta al menos dos pares (energía, potencia)")
    order = np.argsort(power)
    return {
        "potencia_W_kg": power[order],
        "energia_Wh_kg": energy[order],
        "base": basis,
        "avisos": [
            f"la base es «{basis}»: la misma celda por gramo de material "
            "activo y por kilogramo de celda empaquetada difiere en un factor "
            "de tres a cinco, y los gráficos publicados mezclan las dos. Sin "
            "decir la base, el gráfico no se puede comparar con nada"
        ],
    }


# -- capacity fade ------------------------------------------------------

@dataclass
class FadeModel:
    """A fit to capacity against cycle number."""

    model: str
    retention_at_100: Optional[float] = None
    predicted_cycles_to_80: Optional[float] = None
    parameters: dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        text = f"{self.model}: R² = {self.r_squared:.4f}"
        if self.retention_at_100 is not None:
            text += f", {100 * self.retention_at_100:.1f} % a 100 ciclos"
        if self.predicted_cycles_to_80 is not None:
            text += f", 80 % a los {self.predicted_cycles_to_80:.0f} ciclos"
        return text


def fade_model(
    cycles: Sequence[int],
    capacities: Sequence[float],
    model: str = "auto",
) -> FadeModel:
    """Fit capacity loss, and say which mechanism the shape implies.

    Three shapes, and they mean different things:

    ``lineal``
        Loss proportional to cycles. A mechanism that does not stop —
        typically active material being lost.
    ``raiz``
        Loss proportional to √cycles. A film growing and slowing itself
        down as it grows, which is what a solid-electrolyte interphase
        does; extrapolates far better than linear.
    ``exponencial``
        Loss proportional to what is left. Ends at zero.

    ``auto`` fits all three and takes the best, and says so — because
    "80 % capacity after 1000 cycles" extrapolated from 50 cycles depends
    entirely on which of these three was assumed.
    """
    index = np.asarray(cycles, dtype=float)
    capacity = np.asarray(capacities, dtype=float)
    if index.size != capacity.size or index.size < 5:
        raise ValueError("hacen falta al menos cinco ciclos")
    if capacity[0] <= 0:
        raise ValueError("la capacidad inicial tiene que ser positiva")
    retention = capacity / capacity[0]

    candidates = {
        "lineal": (index, lambda x, a, b: a + b * x),
        "raiz": (np.sqrt(index), lambda x, a, b: a + b * np.sqrt(x)),
        "exponencial": (index, None),
    }
    results: dict[str, tuple[float, tuple[float, float]]] = {}
    for name in candidates:
        if name == "exponencial":
            positive = retention > 0
            if np.count_nonzero(positive) < 5:
                continue
            slope, intercept = np.polyfit(index[positive],
                                          np.log(retention[positive]), 1)
            predicted = np.exp(intercept + slope * index)
        else:
            basis = candidates[name][0]
            slope, intercept = np.polyfit(basis, retention, 1)
            predicted = intercept + slope * basis
        total = float(np.sum((retention - np.mean(retention)) ** 2))
        residual = float(np.sum((retention - predicted) ** 2))
        results[name] = (1.0 - residual / total if total > 0 else 0.0,
                         (float(intercept), float(slope)))

    if not results:
        raise ValueError("ningún modelo se pudo ajustar")
    chosen = (model if model != "auto"
              else max(results, key=lambda name: results[name][0]))
    if chosen not in results:
        raise ValueError(
            f"modelo desconocido: {model!r}; hay: {', '.join(sorted(results))}")
    r_squared, (intercept, slope) = results[chosen]

    def predict(cycle: float) -> float:
        if chosen == "lineal":
            return intercept + slope * cycle
        if chosen == "raiz":
            return intercept + slope * math.sqrt(max(cycle, 0.0))
        return math.exp(intercept + slope * cycle)

    at_100 = predict(100.0)
    to_80: Optional[float] = None
    if slope < 0:
        if chosen == "lineal":
            to_80 = (0.8 - intercept) / slope
        elif chosen == "raiz":
            to_80 = ((0.8 - intercept) / slope) ** 2
        else:
            to_80 = (math.log(0.8) - intercept) / slope

    warnings: list[str] = []
    # Best against SECOND best, not best against worst: what decides
    # whether the extrapolation means anything is whether another model
    # describes the measured cycles just as well, not whether some third
    # one describes them badly.
    ranked = sorted((value[0] for value in results.values()), reverse=True)
    gap = ranked[0] - ranked[1] if len(ranked) > 1 else 1.0
    if gap < 0.01:
        warnings.append(
            "los tres modelos ajustan prácticamente igual de bien sobre estos "
            "ciclos: la extrapolación depende entonces por completo de cuál se "
            "elija, y no hay dato para elegir"
        )
    if to_80 and to_80 > 5.0 * float(np.max(index)):
        warnings.append(
            f"extrapolar a {to_80:.0f} ciclos desde {int(np.max(index))} "
            "medidos es más de cinco veces el intervalo medido; la degradación "
            "cambia de mecanismo con el uso y ninguna de estas curvas lo sabe"
        )
    if chosen == "raiz":
        warnings.append(
            "la pérdida va como √ciclos: eso es una capa que crece y se frena "
            "a sí misma al crecer, como una interfase de electrolito sólido"
        )
    elif chosen == "lineal":
        warnings.append(
            "la pérdida es proporcional a los ciclos: un mecanismo que no se "
            "frena, típicamente pérdida de material activo"
        )
    return FadeModel(
        model=chosen, retention_at_100=at_100, predicted_cycles_to_80=to_80,
        parameters={"intercepto": intercept, "pendiente": slope},
        r_squared=r_squared, warnings=warnings,
    )


def koutecky_levich(
    rotation_rpm: Sequence[float],
    limiting_currents_a: Sequence[float],
    area_cm2: float,
    concentration_mol_cm3: float,
    diffusion_cm2_s: float,
    kinematic_viscosity_cm2_s: float = 0.01,
    area_basis: str = "geométrica",
) -> dict[str, object]:
    """Electrons transferred, from a rotating-disk series.

    ``1/j = 1/j_k + 1/(0.62 n F D^{2/3} ν^{−1/6} C ω^{1/2})``. The
    intercept of 1/j against ω^(−1/2) is the kinetic current and the slope
    gives **n**, the number of electrons — which for oxygen reduction is
    the whole point, since 4 means water and 2 means peroxide.

    The n that comes out is only as good as the D and ν that went in, and
    those are tabulated for the electrolyte, not measured. A change of
    10 % in D moves n by 7 %, so an n of 3.7 and an n of 4.0 are not
    distinguishable without knowing them better than that.
    """
    rotation = np.asarray(rotation_rpm, dtype=float)
    currents = np.abs(np.asarray(limiting_currents_a, dtype=float))
    if rotation.size != currents.size or rotation.size < 3:
        raise ValueError("hacen falta al menos tres velocidades de rotación")
    if np.any(currents <= 0):
        raise ValueError("las corrientes límite tienen que ser no nulas")

    omega = rotation * 2.0 * math.pi / 60.0
    x = 1.0 / np.sqrt(omega)
    density = currents / area_cm2
    y = 1.0 / density
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum((y - predicted) ** 2)) / total if total > 0 else 0.0

    levich = (0.62 * FARADAY * diffusion_cm2_s ** (2.0 / 3.0)
              * kinematic_viscosity_cm2_s ** (-1.0 / 6.0)
              * concentration_mol_cm3)
    electrons = 1.0 / (slope * levich) if slope > 0 else None
    kinetic = 1.0 / intercept if intercept > 0 else None

    warnings = _area_warning(area_basis) + [
        "n sale de D y de ν, que son valores tabulados para el electrolito, "
        "no medidos: un 10 % en D mueve n un 7 %, así que un 3.7 y un 4.0 no "
        "se distinguen sin conocerlos mejor que eso"
    ]
    if r_squared < 0.99:
        warnings.append(
            f"R² = {r_squared:.3f} en 1/j frente a ω^(−1/2): las rectas de "
            "Koutecky-Levich tienen que ser muy rectas y paralelas entre "
            "potenciales; si no lo son, el mecanismo cambia con el potencial"
        )
    if electrons and not 0.5 <= electrons <= 6.0:
        warnings.append(
            f"n = {electrons:.2f} está fuera de lo físicamente razonable: "
            "revisa el área, la concentración y las unidades de D"
        )
    return {
        "electrones": electrons,
        "corriente_cinetica_A_cm2": kinetic,
        "pendiente": float(slope),
        "corte": float(intercept),
        "r2": r_squared,
        "avisos": warnings,
    }


__all__ = [
    "FARADAY",
    "GAS_CONSTANT",
    "RANDLES_SEVCIK_25C",
    "DifferentialCapacity",
    "Diffusion",
    "FadeModel",
    "GITTStep",
    "RateCapability",
    "differential_capacity",
    "fade_model",
    "gitt",
    "koutecky_levich",
    "ragone",
    "randles_sevcik",
    "rate_capability",
    "warburg_coefficient",
    "warburg_diffusion",
]
