"""Galvanostatic charge–discharge: capacity, capacitance, energy, cycling.

The constant-current experiment is the one that gives the numbers a device
is actually judged by, and it has one dominant trap: the **IR drop**. At
the instant the current reverses, the potential jumps by ``I·R`` — the
ohmic resistance of the cell — and that jump is not stored energy. Include
it in ``ΔV`` and the capacitance comes out too small; forget to exclude it
from the discharge window and the energy comes out too large. Both errors
appear in the literature, in opposite directions. Here the jump is measured
and removed, and its size is reported, because a large one is the most
useful single diagnostic a charge–discharge curve carries: it says the cell
is limited by resistance and no amount of material optimisation will fix
that.

The second decision here is that **energy is integrated, not assumed**.
``E = ½CV²`` is exact for a capacitor and wrong for everything else, and
the materials this package is aimed at are mostly everything else. The
integral ``E = ∫V dq`` is right in both cases and costs nothing, so the
half-CV-squared value is computed only as a comparison, and the difference
between them is itself a measure of how non-capacitive the electrode is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..core.compat import trapezoid
from .curve import ChargeDischarge, CurveError

#: Points at the start of a branch used to measure the IR jump.
IR_POINTS = 3

#: A coulombic efficiency outside this range is a measurement problem.
EFFICIENCY_LIMITS = (0.5, 1.05)


@dataclass
class Branch:
    """One charge or one discharge."""

    kind: str
    duration_s: float
    v_start: float
    v_end: float
    current_a: float
    ir_drop_v: float
    charge_c: float
    energy_j: float
    """``∫V dq`` over the branch, in joules. Exact for any electrode."""
    capacitance_f: Optional[float]
    linearity: float
    """R² of a straight line through V(t). Near 1 means capacitor-like; a
    plateau gives a low value and means the number in farads is not the
    right way to describe this electrode."""

    @property
    def energy_half_cv2(self) -> Optional[float]:
        if self.capacitance_f is None:
            return None
        return 0.5 * self.capacitance_f * (self.v_start - self.v_end) ** 2

    def __str__(self) -> str:
        text = (
            f"{self.kind:<9s} {self.duration_s:8.2f} s   "
            f"{self.v_start:+.3f} → {self.v_end:+.3f} V   "
            f"I = {1e3 * self.current_a:+.4g} mA   "
            f"q = {1e3 * self.charge_c:.4g} mC"
        )
        if self.ir_drop_v:
            text += f"   caída IR {1e3 * self.ir_drop_v:.1f} mV"
        return text


@dataclass
class GCDResult:
    """The analysis of one charge–discharge measurement."""

    curve: ChargeDischarge
    branches: list[Branch] = field(default_factory=list)
    capacitance_f: Optional[float] = None
    specific_f_per_g: Optional[float] = None
    capacity_c_per_g: Optional[float] = None
    capacity_mah_per_g: Optional[float] = None
    coulombic_efficiency: Optional[float] = None
    energy_efficiency: Optional[float] = None
    energy_wh_per_kg: Optional[float] = None
    power_w_per_kg: Optional[float] = None
    resistance_ohm: Optional[float] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def discharges(self) -> list[Branch]:
        return [b for b in self.branches if b.kind == "descarga"]

    def summary(self) -> str:
        lines = [self.curve.describe(), self.curve.electrode.describe(), ""]
        if self.branches:
            lines.append("Ramas:")
            lines.extend("  " + str(b) for b in self.branches)
        if self.resistance_ohm is not None:
            lines.append("")
            lines.append(
                f"Resistencia interna de la caída IR: {self.resistance_ohm:.4g} Ω"
            )
        if self.capacitance_f is not None:
            lines.append("")
            lines.append(
                f"Capacitancia (descarga, sin la caída IR): "
                f"{1e3 * self.capacitance_f:.4g} mF"
                + (
                    f" = {self.specific_f_per_g:.4g} F/g"
                    if self.specific_f_per_g is not None
                    else ""
                )
            )
        if self.capacity_c_per_g is not None:
            lines.append(
                f"Capacidad: {self.capacity_c_per_g:.4g} C/g = "
                f"{self.capacity_mah_per_g:.4g} mAh/g"
            )
        if self.coulombic_efficiency is not None:
            lines.append("")
            lines.append(
                f"Eficiencia culómbica: {100 * self.coulombic_efficiency:.2f} %"
            )
        if self.energy_efficiency is not None:
            lines.append(
                f"Eficiencia energética : {100 * self.energy_efficiency:.2f} %"
            )
        if self.energy_wh_per_kg is not None:
            lines.append("")
            lines.append(
                f"Energía: {self.energy_wh_per_kg:.4g} Wh/kg    "
                f"Potencia: {self.power_w_per_kg:.4g} W/kg"
            )
            lines.append(
                "  Por masa de material ACTIVO y de un solo electrodo. Un "
                "dispositivo completo lleva dos electrodos, electrolito, "
                "separador y colectores, y su energía por masa total es entre "
                "tres y cinco veces menor. No son comparables"
            )
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


def _linearity(time: np.ndarray, potential: np.ndarray) -> float:
    """R² of a straight line through V(t)."""
    if time.size < 3:
        return 0.0
    coefficients = np.polyfit(time, potential, 1)
    predicted = np.polyval(coefficients, time)
    total = float(((potential - potential.mean()) ** 2).sum())
    if total <= 0.0:
        return 1.0
    return float(1.0 - ((potential - predicted) ** 2).sum() / total)


def _ir_drop(branch: ChargeDischarge, previous_potential: Optional[float]) -> float:
    """The instantaneous jump at the start of a branch, in volts.

    Measured as the difference between the last potential of the previous
    branch and the value the new branch extrapolates back to at its own
    start — a linear extrapolation from the points just after the switch,
    not simply the first point, because the first point also contains the
    beginning of the real response.
    """
    if previous_potential is None or branch.n < IR_POINTS + 2:
        return 0.0
    window = slice(1, 1 + IR_POINTS + 2)
    time = branch.time[window]
    potential = branch.potential[window]
    if time.size < 2:
        return 0.0
    slope, intercept = np.polyfit(time - branch.time[0], potential, 1)
    return float(previous_potential - intercept)


def analyse_gcd(
    curve: ChargeDischarge,
    use_cycle: int = -1,
    window: Optional[tuple[float, float]] = None,
) -> GCDResult:
    """Analyse one charge–discharge measurement.

    Parameters
    ----------
    curve:
        The measurement, one or many cycles.
    use_cycle:
        Which charge/discharge pair to report, ``-1`` for the last.
    window:
        Potential window for the capacitance. Defaults to the discharge's
        own span **after** the IR drop is removed, which is the physically
        meaningful one: the jump is resistance, not stored charge.

    Returns
    -------
    GCDResult
    """
    result = GCDResult(curve=curve)
    segments = curve.segments()
    if not segments:
        result.warnings.append(
            "no se distinguen ramas de carga y descarga por el signo de la "
            "corriente. ¿Está la corriente firmada?"
        )
        return result

    previous: Optional[float] = None
    for kind, branch in segments:
        current = float(np.median(branch.current))
        duration = float(branch.time[-1] - branch.time[0])
        drop = abs(_ir_drop(branch, previous))
        previous = float(branch.potential[-1])

        charge = abs(current) * duration
        # Energy from the integral, valid for any shape of curve.
        charge_axis = np.abs(current) * (branch.time - branch.time[0])
        energy = abs(float(trapezoid(branch.potential, charge_axis)))

        start = float(branch.potential[0])
        end = float(branch.potential[-1])
        effective = abs(end - start) - drop
        capacitance = (
            abs(current) * duration / effective if effective > 1e-9 else None
        )
        result.branches.append(
            Branch(
                kind=kind,
                duration_s=duration,
                v_start=start,
                v_end=end,
                current_a=current,
                ir_drop_v=drop,
                charge_c=charge,
                energy_j=energy,
                capacitance_f=capacitance,
                linearity=_linearity(branch.time, branch.potential),
            )
        )

    discharges = result.discharges
    charges = [b for b in result.branches if b.kind == "carga"]
    if not discharges:
        result.warnings.append("no hay ninguna rama de descarga que analizar")
        return result

    chosen = discharges[use_cycle]
    result.capacitance_f = chosen.capacitance_f
    electrode = curve.electrode
    if chosen.capacitance_f is not None:
        result.specific_f_per_g = electrode.specific(chosen.capacitance_f, "mass")
    result.capacity_c_per_g = electrode.specific(chosen.charge_c, "mass")
    if result.capacity_c_per_g is not None:
        result.capacity_mah_per_g = result.capacity_c_per_g / 3.6

    if chosen.ir_drop_v and abs(chosen.current_a) > 0.0:
        # The jump spans a current change of 2I when going from charge to
        # discharge at the same magnitude.
        result.resistance_ohm = chosen.ir_drop_v / (2.0 * abs(chosen.current_a))

    if charges:
        partner = charges[use_cycle] if len(charges) >= len(discharges) else charges[-1]
        if partner.charge_c > 0:
            result.coulombic_efficiency = chosen.charge_c / partner.charge_c
        if partner.energy_j > 0:
            result.energy_efficiency = chosen.energy_j / partner.energy_j

    per_gram = electrode.specific(chosen.energy_j, "mass")
    if per_gram is not None and chosen.duration_s > 0:
        result.energy_wh_per_kg = per_gram / 3.6
        result.power_w_per_kg = per_gram / chosen.duration_s

    _gcd_warnings(result, chosen)
    return result


def _gcd_warnings(result: GCDResult, discharge: Branch) -> None:
    span = abs(discharge.v_start - discharge.v_end)
    if discharge.ir_drop_v > 0.0 and span > 0.0:
        fraction = discharge.ir_drop_v / span
        if fraction > 0.15:
            result.warnings.append(
                f"la caída IR es el {100 * fraction:.0f} % de la ventana de "
                "descarga. Eso es una celda limitada por resistencia, y ningún "
                "cambio en el material lo arregla: mira el contacto con el "
                "colector, la carga de material y la conductividad del "
                "electrolito. Es además el diagnóstico más útil que da una "
                "curva galvanostática"
            )
    if discharge.linearity < 0.98:
        result.warnings.append(
            f"la descarga no es lineal (R² = {discharge.linearity:.3f}): hay "
            "meseta de potencial. Una meseta es comportamiento de BATERÍA, y "
            "una capacitancia en faradios describe mal a un material así. La "
            "magnitud correcta es la capacidad, en C/g o mAh/g, y está "
            "calculada arriba"
        )
    if result.coulombic_efficiency is not None:
        value = result.coulombic_efficiency
        if value > EFFICIENCY_LIMITS[1]:
            result.warnings.append(
                f"la eficiencia culómbica sale {100 * value:.1f} %, por encima "
                "del 100 %. Físicamente imposible en un ciclo aislado: o hay "
                "una reacción parásita que aporta carga en la descarga, o el "
                "corte de la carga llegó antes de completarse"
            )
        elif value < EFFICIENCY_LIMITS[0]:
            result.warnings.append(
                f"la eficiencia culómbica es {100 * value:.1f} %, muy baja. "
                "Suele significar descomposición del electrolito o "
                "autodescarga apreciable durante la medida"
            )
    if result.energy_wh_per_kg is not None and result.capacitance_f is not None:
        half_cv2 = discharge.energy_half_cv2
        if half_cv2 and discharge.energy_j > 0:
            ratio = half_cv2 / discharge.energy_j
            if abs(ratio - 1.0) > 0.15:
                result.warnings.append(
                    f"½CV² da {100 * ratio:.0f} % de la energía integrada. La "
                    "diferencia mide cuánto se aleja este electrodo de un "
                    "condensador: ½CV² sólo vale si V baja linealmente con la "
                    "carga. La energía que se informa es la integral, que vale "
                    "en los dos casos"
                )


# -- cycling ----------------------------------------------------------


@dataclass
class CyclingResult:
    """Capacity retention over a series of cycles."""

    cycles: list[int] = field(default_factory=list)
    capacity_c: list[float] = field(default_factory=list)
    efficiency: list[float] = field(default_factory=list)
    retention: Optional[float] = None
    fade_per_cycle: Optional[float] = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.cycles:
            return "Sin ciclos que analizar."
        lines = [f"Ciclado: {len(self.cycles)} ciclos"]
        lines.append(
            f"  capacidad inicial {1e3 * self.capacity_c[0]:.4g} mC → "
            f"final {1e3 * self.capacity_c[-1]:.4g} mC"
        )
        if self.retention is not None:
            lines.append(f"  retención: {100 * self.retention:.1f} %")
        if self.fade_per_cycle is not None:
            lines.append(
                f"  pérdida media: {100 * self.fade_per_cycle:.4f} % por ciclo"
            )
        if self.efficiency:
            lines.append(
                f"  eficiencia culómbica media: "
                f"{100 * float(np.mean(self.efficiency)):.2f} %"
            )
        if self.warnings:
            lines.append("")
            lines.extend("  ⚠ " + w for w in self.warnings)
        return "\n".join(lines)


def analyse_cycling(curves: Sequence[ChargeDischarge]) -> CyclingResult:
    """Capacity retention across a list of cycles.

    A single retention percentage says very little on its own: 90 % after
    1000 cycles and 90 % after 50 are different materials. The per-cycle
    fade is reported alongside for that reason, and a retention quoted from
    fewer than ten cycles is flagged as an extrapolation rather than a
    measurement.
    """
    result = CyclingResult()
    for index, curve in enumerate(curves, start=1):
        analysis = analyse_gcd(curve)
        discharges = analysis.discharges
        if not discharges:
            continue
        result.cycles.append(index)
        result.capacity_c.append(discharges[-1].charge_c)
        if analysis.coulombic_efficiency is not None:
            result.efficiency.append(analysis.coulombic_efficiency)

    if len(result.capacity_c) >= 2 and result.capacity_c[0] > 0:
        result.retention = result.capacity_c[-1] / result.capacity_c[0]
        n = len(result.capacity_c) - 1
        result.fade_per_cycle = (1.0 - result.retention) / n
    if len(result.cycles) < 10:
        result.warnings.append(
            f"sólo {len(result.cycles)} ciclos. Una retención medida sobre tan "
            "pocos no se puede extrapolar: la pérdida de los primeros ciclos "
            "es formación, no degradación, y tiene una pendiente distinta"
        )
    return result


__all__ = [
    "EFFICIENCY_LIMITS",
    "IR_POINTS",
    "Branch",
    "CyclingResult",
    "GCDResult",
    "analyse_cycling",
    "analyse_gcd",
]
