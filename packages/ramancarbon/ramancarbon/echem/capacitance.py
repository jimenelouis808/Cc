"""Specific capacitance, measured three ways, and why they disagree.

A supercapacitor paper quotes one number. There are three ways to get it,
they are not the same measurement, and they routinely differ by 20–40 %:

**From a voltammogram**, ``C = ∫I dV / (ν ΔV)`` — the charge the electrode
moves over a potential window, at a stated scan rate. It includes anything
faradaic that happens inside the window, which is why
:func:`~ramancarbon.echem.evaluate.classify_storage` has to decide whether
farads are the right unit at all before this number is allowed out.

**From a charge–discharge curve**, ``C = I Δt / ΔV`` over the discharge
branch with the ohmic drop removed. The most defensible of the three,
because it is the closest thing to how a device is actually used — and the
one most sensitive to where you decide the discharge begins.

**From an impedance spectrum**, ``C′(ω) = −Z″/(ω|Z|²)``. Measured with a
10 mV perturbation around a fixed bias, so it is the small-signal
capacitance of whatever surface the electrolyte reaches *at that frequency*.
It is the highest of the three almost always, because at low current nothing
is rate-limited yet.

They are not competing estimates of one quantity. A device delivers the
GCD number; the EIS number is an upper bound the device never sees; the CV
number sits between them and moves with scan rate. :func:`compare` puts
them side by side on one basis and says which is which, because a paper
that quotes the largest of the three without saying which method produced
it is not reporting a measurement.

The complex-capacitance representation also gives something the other two
cannot: the **relaxation time** ``τ₀ = 1/(2πf₀)`` at the maximum of
``C″(ω)``, the boundary between the frequencies where the device behaves
like a capacitor and those where it behaves like a resistor. It is the
single most useful number an impedance spectrum gives about a
supercapacitor, and it does not depend on knowing the mass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .curve import CurveError, Electrode, Impedance

#: Phase angle in degrees below which the low-frequency end is considered
#: to have reached capacitive behaviour. A perfect capacitor sits at −90°;
#: a porous carbon rarely passes −80°, and above −45° the measurement has
#: not reached the capacitive regime at all and the "capacitance" is mostly
#: resistance.
CAPACITIVE_PHASE_DEG = -70.0


@dataclass
class ComplexCapacitance:
    """``C(ω) = 1/(jωZ)``, split into its real and imaginary parts.

    ``real`` is the capacitance that is actually deliverable at that
    frequency and ``imaginary`` is the dissipative part; the maximum of the
    imaginary part marks the transition between the capacitive and the
    resistive regime.
    """

    frequency: np.ndarray
    real: np.ndarray
    """C′(ω) in farads."""
    imaginary: np.ndarray
    """C″(ω) in farads."""
    series: np.ndarray
    """``−1/(ωZ″)``, the series capacitance. Equal to C′ only where the
    real part of Z is negligible, which is exactly the assumption people
    make without checking when they quote "the capacitance from EIS"."""
    relaxation_s: Optional[float]
    """τ₀ = 1/(2πf₀) at the maximum of C″, or ``None`` if that maximum is
    not inside the measured range."""
    relaxation_frequency_hz: Optional[float] = None
    low_frequency_capacitance: float = 0.0
    """C′ at the lowest measured frequency — the value normally quoted."""
    low_frequency_phase_deg: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        text = (f"C′({self.frequency[0]:.3g} Hz) = "
                f"{1e3 * self.low_frequency_capacitance:.4g} mF")
        if self.relaxation_s:
            text += (f", τ₀ = {self.relaxation_s:.3g} s "
                     f"(f₀ = {self.relaxation_frequency_hz:.3g} Hz)")
        return text


def complex_capacitance(spectrum: Impedance) -> ComplexCapacitance:
    """Complex capacitance of an impedance spectrum.

    ``C(ω) = 1/(jωZ(ω))``, so ``C′ = −Z″/(ω|Z|²)`` and ``C″ = Z′/(ω|Z|²)``.
    Both follow directly from the measured spectrum with no model and no
    fitting, which is what makes this worth doing before any circuit is
    chosen: a circuit is a hypothesis, and C′(ω) is data.

    Notes
    -----
    The sign convention matters here and is easy to get wrong. This package
    stores Z″ with its physical sign — negative for a capacitive response —
    so C′ carries the minus sign explicitly. Code written against a Nyquist
    plot's −Z″ and pasted in here returns a negative capacitance.
    """
    omega = spectrum.omega
    z = spectrum.z
    modulus2 = np.abs(z) ** 2
    real = -z.imag / (omega * modulus2)
    imaginary = z.real / (omega * modulus2)
    with np.errstate(divide="ignore", invalid="ignore"):
        series = np.where(z.imag != 0.0, -1.0 / (omega * z.imag), np.nan)

    # The spectrum is stored descending in frequency, so the last point is
    # the lowest frequency — the one whose capacitance gets quoted.
    order = np.argsort(spectrum.frequency)
    frequency = spectrum.frequency[order]
    real, imaginary, series = real[order], imaginary[order], series[order]
    phase = spectrum.phase_deg[order]

    peak = int(np.argmax(imaginary))
    relaxation: Optional[float] = None
    peak_frequency: Optional[float] = None
    warnings: list[str] = []
    if 0 < peak < imaginary.size - 1:
        peak_frequency = float(frequency[peak])
        relaxation = 1.0 / (2.0 * math.pi * peak_frequency)
    else:
        warnings.append(
            "el máximo de C″ cae en un extremo del intervalo medido, así que "
            "no está dentro de los datos: la constante de tiempo τ₀ no la "
            "determina esta medida. Amplía el barrido de frecuencias hacia "
            + ("las bajas" if peak == 0 else "las altas")
        )

    low_phase = float(phase[0])
    if low_phase > CAPACITIVE_PHASE_DEG:
        warnings.append(
            f"a la frecuencia más baja medida ({frequency[0]:.3g} Hz) el "
            f"ángulo de fase es {low_phase:.0f}°, y un condensador está en "
            "−90°. La medida no ha llegado al régimen capacitivo, así que "
            "esta capacitancia es un límite INFERIOR: baja más la frecuencia"
        )
    if not np.isfinite(series[0]) or series[0] <= 0:
        warnings.append(
            "la capacitancia en serie no se puede calcular a la frecuencia "
            "más baja porque Z″ pasa por cero ahí: el electrodo no es "
            "capacitivo en ese extremo"
        )
    elif series[0] > 1.2 * real[0]:
        warnings.append(
            f"la capacitancia en serie (−1/ωZ″ = {1e3 * series[0]:.3g} mF) y "
            f"la real (C′ = {1e3 * real[0]:.3g} mF) difieren un "
            f"{100 * (series[0] / real[0] - 1):.0f} %: la parte real de la "
            "impedancia no es despreciable a esa frecuencia y las dos "
            "fórmulas dejan de ser la misma. C′ es la que se puede entregar"
        )
    return ComplexCapacitance(
        frequency=frequency, real=real, imaginary=imaginary, series=series,
        relaxation_s=relaxation, relaxation_frequency_hz=peak_frequency,
        low_frequency_capacitance=float(real[0]),
        low_frequency_phase_deg=low_phase, warnings=warnings,
    )


@dataclass
class SpecificCapacitance:
    """One capacitance with its basis, its method and its caveats."""

    farads: float
    method: str
    """``"CV"``, ``"GCD"`` or ``"EIS"``, with the condition in brackets."""
    condition: str
    """The scan rate, current density or frequency it belongs to. A
    capacitance without one is not comparable with anybody else's."""
    per_gram: Optional[float] = None
    per_cm2: Optional[float] = None
    per_cm3: Optional[float] = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.method:<4s} {self.condition:<22s} "
                 f"{1e3 * self.farads:8.4g} mF"]
        if self.per_gram is not None:
            parts.append(f"{self.per_gram:8.4g} F/g")
        if self.per_cm2 is not None:
            parts.append(f"{1e3 * self.per_cm2:8.4g} mF/cm²")
        if self.per_cm3 is not None:
            parts.append(f"{self.per_cm3:8.4g} F/cm³")
        return "  ".join(parts)


def specific(value: float, electrode: Electrode, method: str, condition: str,
             warnings: Optional[list[str]] = None) -> SpecificCapacitance:
    """Normalise one capacitance by every basis the electrode declares."""
    return SpecificCapacitance(
        farads=float(value), method=method, condition=condition,
        per_gram=electrode.specific(value, "mass"),
        per_cm2=electrode.specific(value, "area"),
        per_cm3=electrode.specific(value, "volume"),
        warnings=list(warnings or ()),
    )


def capacitance_from_eis(
    spectrum: Impedance, frequency_hz: Optional[float] = None
) -> SpecificCapacitance:
    """Specific capacitance from an impedance spectrum.

    Parameters
    ----------
    frequency_hz:
        Where to read C′. ``None`` uses the lowest measured frequency,
        which is the convention — and the reason two papers on the same
        material disagree, because one measured down to 10 mHz and the
        other stopped at 100 mHz. The frequency is part of the number and
        is carried in ``condition``.
    """
    analysis = complex_capacitance(spectrum)
    if frequency_hz is None:
        value = analysis.low_frequency_capacitance
        where = float(analysis.frequency[0])
    else:
        target = float(frequency_hz)
        low, high = float(analysis.frequency[0]), float(analysis.frequency[-1])
        if not low <= target <= high:
            raise CurveError(
                f"{target:g} Hz está fuera del intervalo medido "
                f"({low:.3g}–{high:.3g} Hz); no se extrapola una capacitancia"
            )
        value = float(np.interp(target, analysis.frequency, analysis.real))
        where = target
    warnings = list(analysis.warnings)
    warnings.append(
        "una capacitancia de impedancia se mide con una perturbación de unos "
        "10 mV alrededor de un potencial fijo, así que no hay nada limitado "
        "por velocidad: sale alta, y es una cota superior de lo que el "
        "dispositivo entrega, no lo que entrega"
    )
    return specific(value, spectrum.electrode, "EIS",
                    f"C′ a {where:.3g} Hz", warnings)


@dataclass
class CapacitanceComparison:
    """The same electrode measured three ways."""

    entries: list[SpecificCapacitance]
    spread: Optional[float] = None
    """``(max − min)/max`` over the entries, as a fraction."""
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Capacitancia por método:", ""]
        lines.extend("  " + entry.summary() for entry in self.entries)
        if self.spread is not None:
            lines.append("")
            lines.append(f"Dispersión entre métodos: {100 * self.spread:.0f} %")
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + text for text in self.warnings)
        return "\n".join(lines)


def compare(entries: list[SpecificCapacitance]) -> CapacitanceComparison:
    """Put capacitances from different methods side by side.

    The comparison is the point. Three numbers within 10 % of each other
    say the electrode is a real double-layer capacitor and any of them can
    be quoted; three numbers spread over a factor of two say the electrode
    is rate-limited, or faradaic, or that the GCD discharge was measured
    from the wrong point — and then quoting the largest is a choice, not a
    measurement.
    """
    if not entries:
        raise CurveError("no hay ninguna capacitancia que comparar")
    values = [entry.farads for entry in entries if entry.farads > 0]
    warnings: list[str] = []
    spread = None
    if len(values) > 1:
        spread = (max(values) - min(values)) / max(values)
        if spread > 0.30:
            best = max(entries, key=lambda item: item.farads)
            worst = min((e for e in entries if e.farads > 0),
                        key=lambda item: item.farads)
            warnings.append(
                f"los métodos difieren un {100 * spread:.0f} %: "
                f"{best.method} da {1e3 * best.farads:.4g} mF y "
                f"{worst.method} da {1e3 * worst.farads:.4g} mF. No es ruido. "
                "Lo habitual es que la de EIS sea la mayor: se mide con una "
                "perturbación pequeña alrededor de un punto fijo y ahí no hay "
                "nada limitado por velocidad. Entre CV y GCD el orden lo "
                "deciden la velocidad y la corriente que se usaran, y "
                "compararlas solo vale si son equivalentes; si no lo son, "
                "esta diferencia es de las condiciones y no del electrodo"
            )
    methods = {entry.method for entry in entries}
    if "GCD" in methods and len(methods) > 1:
        warnings.append(
            "el número que entrega un dispositivo es el de GCD. Los otros dos "
            "son diagnósticos: útiles para entender el electrodo, no para "
            "anunciar su prestación"
        )
    if not any(entry.per_gram is not None for entry in entries):
        warnings.append(
            "sin masa activa no hay F/g, que es como se publica casi todo. "
            "Ponla en el electrodo: es la masa de material ACTIVO, no la del "
            "electrodo entero"
        )
    return CapacitanceComparison(entries=entries, spread=spread,
                                 warnings=warnings)


__all__ = [
    "CAPACITIVE_PHASE_DEG",
    "CapacitanceComparison",
    "ComplexCapacitance",
    "SpecificCapacitance",
    "capacitance_from_eis",
    "compare",
    "complex_capacitance",
    "specific",
]
