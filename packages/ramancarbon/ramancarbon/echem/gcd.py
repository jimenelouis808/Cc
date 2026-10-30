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

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..core.compat import trapezoid
from .curve import ChargeDischarge

#: Points at the start of a branch used to measure the IR jump.
IR_POINTS = 3

#: A coulombic efficiency outside this range is a measurement problem.
EFFICIENCY_LIMITS = (0.5, 1.05)

#: Fewest points a half cycle needs to be worth splitting out at all.
#:
#: Two: a start and an end. That is enough for the charge ``I·Δt`` and for
#: the window ``ΔV``, which together are the capacitance everybody quotes,
#: and it is not enough for anything that needs the SHAPE of the branch.
#: Those refuse separately and say so, which is the point of letting a
#: short branch through instead of dropping it: a real GCPL saves on a
#: potential or time condition and routinely records three points per half
#: cycle, so a floor set for the benefit of the shape analysis throws away
#: the whole measurement to protect a number nobody asked for.
MIN_HALFCYCLE_POINTS = 2

#: Fewest points before the SHAPE of a branch means anything.
#:
#: Below this the slope fit, the energy integral and the linearity are not
#: measurements of the branch, they are measurements of its two endpoints.
MIN_SHAPE_POINTS = 12

#: Relative difference between the charge and discharge current magnitudes
#: above which the pair is not a symmetric cycle.
#:
#: Two per cent is below any deliberate setting and above the resolution
#: with which a potentiostat reports its own set point.
ASYMMETRY_TOLERANCE = 0.02

#: What to do with a half cycle whose charge and discharge currents differ.
#:
#: ``"aparte"`` analyses it and reports it, but keeps it out of the rate
#: curve and the Ragone plot, because those have one current per point and
#: this one has two. ``"excluir"`` drops it. ``"interpolar"`` places it on
#: the discharge current and ALSO reports what the symmetric points
#: interpolate to there, so the two can be compared. ``"incluir"`` takes it
#: at face value, which is the caller overriding this whole paragraph.
ASYMMETRY_POLICIES = ("aparte", "excluir", "interpolar", "incluir")


@dataclass
class Halfcycle:
    """One charge or one discharge, as MEASURED: three arrays and a label.

    This is deliberately NOT a :class:`~ramancarbon.echem.curve.ChargeDischarge`.
    That class is a loaded measurement and refuses fewer than ten points,
    which is right for a file and wrong for a half cycle: a real GCPL
    records three. Splitting into curves therefore raised ``CurveError``
    from inside the splitter on every one of the user's five files, so the
    galvanostatic analysis could not run on real galvanostatic data at
    all.
    """

    kind: str
    """``"carga"`` or ``"descarga"``."""
    index: int
    """Position of this half cycle in the file, counting from zero."""
    time: np.ndarray
    potential: np.ndarray
    current: np.ndarray

    @property
    def n(self) -> int:
        return int(self.time.size)

    @property
    def cycle(self) -> int:
        """Which full cycle this half belongs to, counting from zero."""
        return self.index // 2

    @property
    def current_a(self) -> float:
        """The median current, in amps, sign included."""
        return float(np.median(self.current))

    @property
    def duration_s(self) -> float:
        return float(self.time[-1] - self.time[0])

    @property
    def floor_v(self) -> float:
        """The branch's own LOWER potential end.

        Energy is referenced to this and not to zero. On a three-electrode
        axis zero is wherever the reference electrode sits, so an energy
        integrated from there is not an energy anybody can recover — and
        with a window that straddles zero it is not even positive.
        """
        return min(float(self.potential[0]), float(self.potential[-1]))


def half_cycles(curve: ChargeDischarge,
                min_points: int = MIN_HALFCYCLE_POINTS) -> list[Halfcycle]:
    """Split a measurement into half cycles by the SIGN of the current.

    The sign is the only thing that separates a charge from a discharge,
    which is why a charge-discharge curve is useless without its current
    column. Runs shorter than ``min_points`` are dropped as switching
    transients.
    """
    current = np.asarray(curve.current, dtype=float)
    sign = np.sign(current)
    sign[sign == 0] = 1.0
    breaks = np.flatnonzero(np.diff(sign) != 0) + 1
    bounds = [0, *breaks.tolist(), current.size]
    time = np.asarray(curve.time, dtype=float)
    potential = np.asarray(curve.potential, dtype=float)
    out: list[Halfcycle] = []
    for position in range(len(bounds) - 1):
        start, stop = bounds[position], bounds[position + 1]
        if stop - start < max(2, min_points):
            continue
        where = slice(start, stop)
        out.append(
            Halfcycle(
                kind="carga" if sign[start] > 0 else "descarga",
                index=len(out),
                time=time[where],
                potential=potential[where],
                current=current[where],
            )
        )
    return out


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
    """``I·Δt/ΔV`` over the branch's own window, the IR jump removed. The
    number everybody quotes, and the one that is wrong first when the
    curve is not a straight line."""
    capacitance_slope_f: Optional[float] = None
    """``I/|dV/dt|`` from a straight line through the MIDDLE of the
    branch. On a curved discharge this is the local capacitance where the
    curve is straightest, which is a different question from the average
    over the window and usually a smaller answer."""
    capacitance_energy_f: Optional[float] = None
    """``2E/ΔV²`` with ``E`` the energy actually delivered down to the
    lower limit. This is the capacitance an ideal capacitor would need to
    store the SAME ENERGY over the same window, and it is the one that
    should be quoted for a pseudocapacitor: the ΔV method assumes the
    discharge is linear and over-reports when it is not, by exactly the
    amount the plateau bends."""
    linearity: float = 1.0
    """R² of a straight line through V(t). Near 1 means capacitor-like; a
    plateau gives a low value and means the number in farads is not the
    right way to describe this electrode."""
    index: int = 0
    """Position of this half cycle in the file, counting from zero."""
    cycle: int = 0
    """Which full cycle it belongs to, counting from zero."""
    points: int = 0
    """How many points the branch has. Below :data:`MIN_SHAPE_POINTS` the
    shape-based numbers are refused rather than computed from endpoints."""
    floor_v: float = 0.0
    """The potential the energy is referenced to: the branch's own lower
    end, not zero."""
    partner_current_a: Optional[float] = None
    """The current of the half cycle this one was switched FROM. The jump
    at the switch spans the change in current, which is the sum of the two
    magnitudes and only equals ``2|I|`` when they are equal."""
    refusals: list[str] = field(default_factory=list)
    """Why a number is missing, when it is missing for a reason."""

    @property
    def asymmetric(self) -> bool:
        """Whether this branch and the one before it ran at different |I|."""
        if self.partner_current_a is None:
            return False
        mine, theirs = abs(self.current_a), abs(self.partner_current_a)
        if max(mine, theirs) <= 0.0:
            return False
        return abs(mine - theirs) / max(mine, theirs) > ASYMMETRY_TOLERANCE

    @property
    def current_step_a(self) -> Optional[float]:
        """The change in current across the switch into this branch."""
        if self.partner_current_a is None:
            return None
        return abs(self.current_a) + abs(self.partner_current_a)

    def capacitances(self) -> dict[str, Optional[float]]:
        """The three conventions by name, for reporting side by side."""
        return {
            "ΔV (I·Δt/ΔV)": self.capacitance_f,
            "pendiente (I/|dV/dt|)": self.capacitance_slope_f,
            "energía (2E/ΔV²)": self.capacitance_energy_f,
        }

    @property
    def capacitance_spread(self) -> Optional[float]:
        """``(max−min)/max`` over the three, or ``None``.

        The disagreement is the diagnostic. Three numbers within a few
        per cent say the discharge really is a straight line and any of
        them can be quoted; a spread of thirty per cent says it is not,
        and then choosing the largest is a choice rather than a
        measurement.
        """
        values = [v for v in self.capacitances().values() if v and v > 0]
        if len(values) < 2:
            return None
        return (max(values) - min(values)) / max(values)

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
    capacitance_slope_f: Optional[float] = None
    capacitance_energy_f: Optional[float] = None
    capacitance_spread: Optional[float] = None
    """``(max−min)/max`` over the three conventions. The disagreement is
    the diagnostic: it measures how far the discharge is from a straight
    line, which is the one thing a single capacitance cannot say."""
    specific_by_method: dict[str, Optional[float]] = field(default_factory=dict)
    """F/g by convention name, for the report and the GUI."""
    specific_f_per_g: Optional[float] = None
    capacity_c_per_g: Optional[float] = None
    capacity_mah_per_g: Optional[float] = None
    coulombic_efficiency: Optional[float] = None
    energy_efficiency: Optional[float] = None
    energy_wh_per_kg: Optional[float] = None
    power_w_per_kg: Optional[float] = None
    resistance_ohm: Optional[float] = None
    resistance_how: str = ""
    """How the internal resistance was obtained, the current step included.
    A resistance divided by the wrong step is the single easiest number to
    get twice too large here."""
    max_power_w_per_kg: Optional[float] = None
    """``ΔV²/(4·ESR)`` per kilogram of active material: the most a cell of
    this resistance can deliver into a matched load. It is an upper bound
    on the Ragone plot's power axis, not a measured point."""
    charge_current_a: Optional[float] = None
    """The current of the charge that preceded the reported discharge."""
    asymmetric: bool = False
    """Whether the reported pair ran at two different current magnitudes.
    Such a point has no single current density and stays out of the rate
    curve and the Ragone plot unless the caller overrides it."""
    leakage_a: Optional[float] = None
    """``(q_dis − q_ch)/t``: the parasitic current the charge balance
    implies. Positive means charge comes OUT that did not go in, which is
    a reduction reaction, not storage."""
    specific_uf_per_cm2: Optional[float] = None
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


def _ir_drop(branch, previous_potential: Optional[float]) -> float:
    """The instantaneous jump at the start of a branch, in volts.

    Measured as the difference between the last potential of the previous
    branch and the value the new branch extrapolates back to at its own
    start — a linear extrapolation from the points just after the switch,
    not simply the first point, because the first point also contains the
    beginning of the real response.

    On a branch too short to extrapolate the first point IS the estimate,
    which is a little small by whatever the real response did in one
    sample. Refusing instead would throw the jump away on exactly the
    files that have it: a GCPL saving three points per half cycle records
    the switch and nothing else, and on the user's five files the
    first-point estimate returns 77, 80, 80, 80 and 79 ohm — consistent to
    four per cent across a factor of eight in current, which is as much as
    the extrapolation would buy.
    """
    if previous_potential is None or branch.n < 2:
        return 0.0
    if branch.n >= IR_POINTS + 2:
        window = slice(1, 1 + IR_POINTS + 2)
        time = np.asarray(branch.time[window], dtype=float)
        potential = np.asarray(branch.potential[window], dtype=float)
        if time.size >= 2 and np.ptp(time) > 0.0:
            _, intercept = np.polyfit(time - float(branch.time[0]), potential, 1)
            return float(previous_potential - intercept)
    return float(previous_potential - float(branch.potential[0]))


#: Fraction of a branch trimmed from each end before the slope fit.
#:
#: The IR jump lives at the start and the curve bends hard at the end of
#: a real discharge; a straight line through either is not the slope of
#: anything.
SLOPE_TRIM = 0.15


def _slope_capacitance(branch, current: float) -> Optional[float]:
    """``I/|dV/dt|`` from the straightest part of the branch."""
    time = np.asarray(branch.time, dtype=float)
    potential = np.asarray(branch.potential, dtype=float)
    if time.size < 12 or abs(current) <= 0.0:
        return None
    trim = max(1, int(round(SLOPE_TRIM * time.size)))
    inner_t, inner_v = time[trim:-trim], potential[trim:-trim]
    if inner_t.size < 5 or np.ptp(inner_t) <= 0:
        return None
    slope = float(np.polyfit(inner_t, inner_v, 1)[0])
    if abs(slope) < 1e-12:
        return None
    return abs(current) / abs(slope)


def _energy_capacitance(branch, current: float,
                        effective: float) -> Optional[float]:
    """``2E/ΔV²`` with ``E`` the energy delivered down to the lower limit.

    Referenced to the branch's own end potential rather than to zero,
    because zero on a three-electrode axis is wherever the reference
    electrode happens to sit and an energy measured from there is not an
    energy anybody can recover.
    """
    time = np.asarray(branch.time, dtype=float)
    potential = np.asarray(branch.potential, dtype=float)
    if time.size < 4 or effective <= 1e-9 or abs(current) <= 0.0:
        return None
    floor = float(potential[-1])
    above = potential - floor
    energy = abs(current) * abs(float(trapezoid(above, time)))
    if energy <= 0.0:
        return None
    return 2.0 * energy / (effective ** 2)


#: Relative drop between two consecutive cycles that is an EVENT rather
#: than degradation.
#:
#: Degradation is gradual by definition: a mechanism that removes a
#: third of the capacity between one cycle and the next is not wear, it
#: is something coming loose. Twenty per cent is far above the
#: cycle-to-cycle scatter of any of the user's files and far below the
#: 66 % step their 5 uA file takes at cycle 319.
CYCLE_STEP_LIMIT = 0.20


#: Spread between the three capacitance conventions that counts as
#: disagreement rather than rounding.
#:
#: On an ideal capacitor the three agree to about one per cent, measured
#: on a synthetic 50 mF discharge; on a plateau they came out 49 and
#: 36 mF, a spread of 28 %. Ten is comfortably above the first and well
#: below the second.
CAPACITANCE_SPREAD_LIMIT = 0.10


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
    halves = half_cycles(curve)
    if not halves:
        result.warnings.append(
            "no se distinguen ramas de carga y descarga por el signo de la "
            "corriente. ¿Está la corriente firmada?"
        )
        return result

    previous_potential: Optional[float] = None
    previous_current: Optional[float] = None
    for half in halves:
        current = half.current_a
        duration = half.duration_s
        drop = abs(_ir_drop(half, previous_potential))

        start = float(half.potential[0])
        end = float(half.potential[-1])
        effective = abs(end - start) - drop
        charge = abs(current) * duration
        capacitance = (
            abs(current) * duration / effective if effective > 1e-9 else None
        )

        # Energy referenced to the branch's OWN lower end, never to zero.
        # On a three-electrode axis zero is wherever the reference
        # electrode sits, so an energy integrated from there is not an
        # energy anybody can recover; and over a window that straddles
        # zero the integral is not even positive. The user's window is
        # -0.6 to +0.4 V, where the absolute-axis integral comes out at a
        # fifth of the real one.
        floor = half.floor_v
        charge_axis = abs(current) * (half.time - half.time[0])
        energy = abs(float(trapezoid(half.potential - floor, charge_axis)))

        refusals: list[str] = []
        if half.n < MIN_SHAPE_POINTS:
            refusals.append(
                f"la rama tiene {half.n} puntos: por debajo de "
                f"{MIN_SHAPE_POINTS} no hay FORMA que medir, así que la "
                "pendiente, la linealidad y la capacitancia por energía no "
                "se informan. La carga y la capacitancia por ΔV sí, que "
                "sólo necesitan los extremos"
            )

        shaped = half.n >= MIN_SHAPE_POINTS
        result.branches.append(
            Branch(
                kind=half.kind,
                duration_s=duration,
                v_start=start,
                v_end=end,
                current_a=current,
                ir_drop_v=drop,
                charge_c=charge,
                energy_j=energy,
                capacitance_f=capacitance,
                capacitance_slope_f=(
                    _slope_capacitance(half, current) if shaped else None),
                capacitance_energy_f=(
                    _energy_capacitance(half, current, effective)
                    if shaped else None),
                linearity=(
                    _linearity(half.time, half.potential) if shaped else 0.0),
                index=half.index,
                cycle=half.cycle,
                points=half.n,
                floor_v=floor,
                partner_current_a=previous_current,
                refusals=refusals,
            )
        )
        previous_potential = end
        previous_current = current

    discharges = result.discharges
    charges = [b for b in result.branches if b.kind == "carga"]
    if not discharges:
        result.warnings.append("no hay ninguna rama de descarga que analizar")
        return result

    chosen = discharges[use_cycle]
    result.capacitance_f = chosen.capacitance_f
    result.capacitance_slope_f = chosen.capacitance_slope_f
    result.capacitance_energy_f = chosen.capacitance_energy_f
    electrode = curve.electrode
    if chosen.capacitance_f is not None:
        result.specific_f_per_g = electrode.specific(chosen.capacitance_f, "mass")
    result.specific_by_method = {
        name: (electrode.specific(value, "mass") if value else None)
        for name, value in chosen.capacitances().items()
    }
    spread = chosen.capacitance_spread
    result.capacitance_spread = spread
    if spread is not None and spread > CAPACITANCE_SPREAD_LIMIT:
        best = max((v for v in chosen.capacitances().values() if v), default=0.0)
        worst = min((v for v in chosen.capacitances().values() if v), default=0.0)
        result.warnings.append(
            f"los tres convenios de capacitancia difieren un "
            f"{100 * spread:.0f} % ({1e3 * best:.4g} mF frente a "
            f"{1e3 * worst:.4g} mF). Eso NO es ruido: es que la descarga no "
            f"es una recta (R² = {chosen.linearity:.3f}). El de ΔV supone "
            "que sí lo es y sobreinforma exactamente lo que la meseta se "
            "curva; para un pseudocondensador el defendible es el de "
            "ENERGÍA, que es la capacitancia que almacenaría la misma "
            "energía en la misma ventana. Elegir el mayor es una elección, "
            "no una medida"
        )
    result.capacity_c_per_g = electrode.specific(chosen.charge_c, "mass")
    if result.capacity_c_per_g is not None:
        result.capacity_mah_per_g = result.capacity_c_per_g / 3.6

    # The jump at the switch spans the CHANGE in current, which is
    # |I_charge| + |I_discharge|. That equals 2|I| only when the two
    # magnitudes are equal, and the user's fifth file charges at 60 uA and
    # discharges at 20: there the old denominator returns 158 ohm for a
    # cell whose other four files all agree on 80. Measured across the
    # five: 76.7, 80.0, 80.1, 79.8 and 79.3 ohm with the sum, against
    # 158.5 for the asymmetric one with 2|I|. The asymmetric file agreeing
    # with the other four is the check, and it is a measurement rather
    # than an argument.
    step = chosen.current_step_a
    if chosen.ir_drop_v and step and step > 0.0:
        result.resistance_ohm = chosen.ir_drop_v / step
        result.resistance_how = (
            f"caída IR {1e3 * chosen.ir_drop_v:.3f} mV entre un escalón de "
            f"corriente de {1e6 * step:.3f} µA "
            f"(|{1e6 * abs(chosen.partner_current_a or 0.0):.3f}| + "
            f"|{1e6 * abs(chosen.current_a):.3f}| µA)"
        )
    elif chosen.ir_drop_v and abs(chosen.current_a) > 0.0:
        # No partner: this discharge is the first branch in the file, so
        # there was no switch and no measurable jump to divide.
        result.warnings.append(
            "la descarga informada es la primera rama del archivo, así que "
            "no hay cambio de corriente antes de ella y de su salto no sale "
            "ninguna resistencia"
        )

    # The charge that pairs with a discharge is the one it was switched
    # FROM, which is the branch immediately before it. Indexing a separate
    # list of charges by the same position happens to be right when the
    # file starts with a charge and is wrong as soon as it does not.
    partner = None
    for candidate in result.branches:
        if candidate.index == chosen.index - 1 and candidate.kind == "carga":
            partner = candidate
            break
    if partner is None and charges:
        partner = charges[-1]
        result.warnings.append(
            "la descarga informada no va precedida de una carga, así que la "
            "eficiencia se calcula contra la última carga del archivo y no "
            "contra su pareja"
        )
    if partner is not None:
        if partner.charge_c > 0:
            result.coulombic_efficiency = chosen.charge_c / partner.charge_c
        if partner.energy_j > 0:
            result.energy_efficiency = chosen.energy_j / partner.energy_j
        result.charge_current_a = partner.current_a
        if chosen.asymmetric:
            result.asymmetric = True
            result.warnings.append(
                f"la carga va a {1e6 * abs(partner.current_a):.3g} µA y la "
                f"descarga a {1e6 * abs(chosen.current_a):.3g} µA. **No es "
                "un ciclo simétrico**: la eficiencia culómbica compara dos "
                "corrientes distintas y no mide reversibilidad, y este punto "
                "no tiene UNA densidad de corriente, así que no se puede "
                "poner en una curva de capacidad frente a corriente ni en un "
                "Ragone sin decir cuál de las dos se usa. La resistencia sí "
                "sale bien, porque su escalón es la SUMA de las dos"
            )

    # Both per KILOGRAM, and the factor of a thousand is the whole reason
    # this is spelled out. `specific` normalises by the mass in GRAMS, so
    # it returns J/g; the energy conversion folds the thousand into the
    # 3.6 (1 J/g = 1000 J/kg = 1000/3600 Wh/kg) and the power one has to
    # carry it explicitly. Dividing J/g by seconds gives W/g, and
    # reporting that as W/kg put every Ragone plot this program drew a
    # factor of a thousand to the left. Checked against an ideal 1 F
    # capacitor: 1 V, 1 A, 1 g gives 0.5 J in 1 s, which is 0.1389 Wh/kg
    # and 500 W/kg.
    per_gram = electrode.specific(chosen.energy_j, "mass")
    if per_gram is not None and chosen.duration_s > 0:
        result.energy_wh_per_kg = per_gram / 3.6
        result.power_w_per_kg = 1000.0 * per_gram / chosen.duration_s

    # The parasitic current the charge balance implies. A coulombic
    # efficiency above 100 % is charge coming out that did not go in, and
    # dividing the excess by the time turns an impossible percentage into
    # a current that can be compared across the series: a leak is a
    # current, so the SAME leak shows up as a big excess at a small
    # current and a small one at a large current. The user's four
    # symmetric files run 9.65, 5.13, 2.06 and 0.98 % at 5, 10, 20 and
    # 40 uA, a factor of eight in current, and come back as one current.
    #
    # The denominator is the WHOLE cycle, charge plus discharge, because
    # a parasitic reaction runs the whole time and not only while the
    # cell is being charged. The charge alone looks equivalent on a
    # symmetric cycle -- the two halves are the same length -- and is
    # decisively wrong on an asymmetric one, which is how the choice got
    # settled. Over the five files:
    #
    #   denominator      5 uA   10 uA   20 uA   40 uA   60/-20 uA
    #   charge only     0.534   0.448   0.390   0.383     0.915
    #   whole cycle     0.254   0.219   0.193   0.190     0.226
    #
    # With the charge alone the asymmetric file sits 2.4x above the rest;
    # with the whole cycle it lands in the middle of them. All five
    # agreeing is the check.
    if partner is not None and partner.charge_c > 0:
        period = partner.duration_s + chosen.duration_s
        if period > 0:
            excess = chosen.charge_c - partner.charge_c
            result.leakage_a = excess / period

    if chosen.capacitance_f is not None:
        per_area = electrode.specific(chosen.capacitance_f, "area")
        if per_area is not None:
            result.specific_uf_per_cm2 = 1e6 * per_area

    # The matched-load ceiling. It belongs with the resistance and not
    # with the Ragone points, because no measured discharge reaches it.
    if result.resistance_ohm and result.resistance_ohm > 0.0:
        span = abs(chosen.v_start - chosen.v_end) - chosen.ir_drop_v
        if span > 0.0:
            ceiling = span ** 2 / (4.0 * result.resistance_ohm)
            per_gram_power = electrode.specific(ceiling, "mass")
            if per_gram_power is not None:
                result.max_power_w_per_kg = 1000.0 * per_gram_power

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
            text = (
                f"la eficiencia culómbica sale {100 * value:.1f} %, por encima "
                "del 100 %. Físicamente imposible en un ciclo aislado: o hay "
                "una reacción parásita que aporta carga en la descarga, o el "
                "corte de la carga llegó antes de completarse"
            )
            if result.leakage_a:
                text += (
                    f". El exceso equivale a {1e6 * result.leakage_a:+.3g} µA "
                    "de corriente parásita, que es la forma de comparar este "
                    "número entre corrientes distintas: una fuga es una "
                    "corriente, así que la MISMA fuga sale como un exceso "
                    "grande a corriente pequeña y pequeño a corriente grande"
                )
            result.warnings.append(text)
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
    capacitance_f: list[float] = field(default_factory=list)
    """``I·Δt/ΔV`` per cycle. On a file whose half cycles are three points
    long this is the only capacitance there is, and it is the standard
    one."""
    current_a: list[float] = field(default_factory=list)
    """The discharge current of each cycle, so a file that changes current
    part way through can still be read."""
    retention: Optional[float] = None
    fade_per_cycle: Optional[float] = None
    source: str = ""
    """Where the cycles came from: one multi-cycle measurement, or a list
    of separate ones."""
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


def cycling_from_curve(curve: ChargeDischarge) -> CyclingResult:
    """Capacity, capacitance and efficiency cycle by cycle, from ONE file.

    This is the analysis the galvanostatic section did not have.
    :func:`analyse_cycling` takes a LIST of measurements and produces one
    point per measurement, which is the right shape when every cycle was
    saved to its own file and the wrong one for a GCPL: the user's five
    files hold **1001 cycles each**, and the whole of that — half the
    capacitance lost over the first file, a per-cycle coulombic excess
    that turns out to be a constant leak — was unreachable, because each
    file collapsed to a single number.

    Each cycle is a charge followed by its own discharge. Capacitance is
    ``I·Δt/ΔV`` with the IR jump removed, which needs only the two
    endpoints of the branch and so survives the three-point half cycles a
    real GCPL writes.
    """
    result = CyclingResult(source=f"un solo archivo: {curve.name}")
    halves = half_cycles(curve)
    if not halves:
        result.warnings.append(
            "no se distinguen ramas de carga y descarga por el signo de la "
            "corriente"
        )
        return result

    previous_end: Optional[float] = None
    pending_charge: Optional[Halfcycle] = None
    number = 0
    for half in halves:
        drop = abs(_ir_drop(half, previous_end))
        previous_end = float(half.potential[-1])
        if half.kind == "carga":
            pending_charge = half
            continue
        # A discharge closes a cycle.
        span = abs(float(half.potential[-1]) - float(half.potential[0])) - drop
        charge = abs(half.current_a) * half.duration_s
        if span <= 1e-9 or charge <= 0.0:
            continue
        number += 1
        result.cycles.append(number)
        result.capacity_c.append(charge)
        result.capacitance_f.append(charge / span)
        result.current_a.append(half.current_a)
        if pending_charge is not None:
            given = abs(pending_charge.current_a) * pending_charge.duration_s
            if given > 0.0:
                result.efficiency.append(charge / given)
        pending_charge = None

    _finish_cycling(result)
    return result


def _finish_cycling(result: CyclingResult) -> None:
    """Retention, fade and the warnings both cycling routes share."""
    if len(result.capacity_c) >= 2 and result.capacity_c[0] > 0:
        result.retention = result.capacity_c[-1] / result.capacity_c[0]
        n = len(result.capacity_c) - 1
        result.fade_per_cycle = (1.0 - result.retention) / n
    if 0 < len(result.cycles) < 10:
        result.warnings.append(
            f"sólo {len(result.cycles)} ciclos. Una retención medida sobre tan "
            "pocos no se puede extrapolar: la pérdida de los primeros ciclos "
            "es formación, no degradación, y tiene una pendiente distinta"
        )
    if result.current_a:
        magnitudes = np.abs(np.asarray(result.current_a, dtype=float))
        if magnitudes.max() > 0 and (
                magnitudes.max() - magnitudes.min()) / magnitudes.max() > ASYMMETRY_TOLERANCE:
            result.warnings.append(
                f"la corriente de descarga no es la misma en todos los ciclos "
                f"({1e6 * magnitudes.min():.3g} a {1e6 * magnitudes.max():.3g} "
                "µA), así que esta curva mezcla degradación con cambio de "
                "velocidad y su retención no es una retención"
            )
    # A sudden step is not a fade, and a fade model fitted across one
    # describes the step rather than a mechanism. Measured on the user's
    # 5 uA file: 1001 cycles with exactly two drops over 20 % -- the
    # first cycle, which is formation, and cycle 319, which loses 66 %
    # in one cycle (298.7 to 99.9 uF) while the coulombic efficiency
    # barely moves. That is material coming off the disc or a contact
    # changing, not degradation, and `fade_model` happily returns a
    # square-root law for it.
    if len(result.capacitance_f) > 2:
        values = np.asarray(result.capacitance_f, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            change = np.diff(values) / values[:-1]
        change = np.where(np.isfinite(change), change, 0.0)
        # The first cycle is formation and is expected to move.
        steps = np.flatnonzero(change[1:] < -CYCLE_STEP_LIMIT) + 1
        for position in steps[:3]:
            result.warnings.append(
                f"entre el ciclo {result.cycles[position]} y el siguiente la "
                f"capacitancia cae un {100 * abs(change[position]):.0f} % "
                f"({1e6 * values[position]:.4g} a "
                f"{1e6 * values[position + 1]:.4g} µF). Un salto NO es una "
                "degradación: la degradación es gradual por definición, y "
                "esto es material que se desprende, un contacto que cambia o "
                "una burbuja. La retención medida de punta a punta lo "
                "incluye, y un modelo de desvanecimiento ajustado a través "
                "de él describe el salto, no un mecanismo"
            )
    if result.efficiency:
        mean = float(np.mean(result.efficiency))
        if mean > EFFICIENCY_LIMITS[1]:
            result.warnings.append(
                f"la eficiencia culómbica media es {100 * mean:.2f} %, por "
                "encima del 100 % en el promedio de todos los ciclos. Eso no "
                "es ruido de un ciclo: es carga que sale sin haber entrado, o "
                "sea una reacción parásita sostenida durante toda la medida"
            )


def analyse_cycling(curves: Sequence[ChargeDischarge]) -> CyclingResult:
    """Capacity retention across a list of cycles.

    A single retention percentage says very little on its own: 90 % after
    1000 cycles and 90 % after 50 are different materials. The per-cycle
    fade is reported alongside for that reason, and a retention quoted from
    fewer than ten cycles is flagged as an extrapolation rather than a
    measurement.
    """
    result = CyclingResult(source=f"{len(curves)} medidas, una por ciclo")
    for index, curve in enumerate(curves, start=1):
        analysis = analyse_gcd(curve)
        discharges = analysis.discharges
        if not discharges:
            continue
        chosen = discharges[-1]
        result.cycles.append(index)
        result.capacity_c.append(chosen.charge_c)
        result.current_a.append(chosen.current_a)
        if chosen.capacitance_f is not None:
            result.capacitance_f.append(chosen.capacitance_f)
        if analysis.coulombic_efficiency is not None:
            result.efficiency.append(analysis.coulombic_efficiency)

    _finish_cycling(result)
    return result


__all__ = [
    "ASYMMETRY_POLICIES",
    "CYCLE_STEP_LIMIT",
    "ASYMMETRY_TOLERANCE",
    "EFFICIENCY_LIMITS",
    "IR_POINTS",
    "MIN_HALFCYCLE_POINTS",
    "MIN_SHAPE_POINTS",
    "Branch",
    "CyclingResult",
    "GCDResult",
    "Halfcycle",
    "analyse_cycling",
    "analyse_gcd",
    "cycling_from_curve",
    "half_cycles",
]
