"""Electrochemistry figures. Written against matplotlib axes, never Tk.

Two conventions are followed because the field follows them, and departing
from either makes a plot harder to read for the people who will read it:
a Nyquist plot shows **−Z″** against Z′ with **equal aspect**, and a
Ragone plot is logarithmic on both axes. The equal aspect is not cosmetic —
a semicircle only looks like a semicircle at aspect 1, and a stretched one
invites the eye to read a depression angle that is not there.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..echem.curve import ChargeDischarge, Impedance, Voltammogram
from ..echem.cv import RateStudy
from ..echem.eis import EISResult
from ..echem.evaluate import CatalysisResult
from ..echem.gcd import GCDResult
from .base import placeholder
from .theme import Palette


def plot_cv(
    ax,
    curves: Sequence[Voltammogram],
    palette: Palette,
    peaks: Optional[Sequence] = None,
    normalise_by_rate: bool = False,
) -> None:
    """One or several voltammograms.

    ``normalise_by_rate`` divides the current by the scan rate, which
    superimposes the curves of a pure capacitor exactly. That makes the
    plot a test rather than a picture: whatever fails to superimpose is
    the part of the response that is not capacitive.
    """
    for index, curve in enumerate(curves):
        current = curve.current / curve.scan_rate if normalise_by_rate else curve.current
        scale = 1.0 if normalise_by_rate else 1e3
        ax.plot(
            curve.potential,
            scale * current,
            color=palette.component_colour(index) if len(curves) > 1 else palette.data,
            linewidth=1.1,
            label=f"{1e3 * curve.scan_rate:g} mV/s",
        )
    if peaks:
        for peak in peaks:
            ax.plot(
                [peak.potential], [1e3 * peak.current], "v",
                color=palette.fitted, markersize=6,
            )
            ax.annotate(
                f"{peak.potential:+.3f}",
                xy=(peak.potential, 1e3 * peak.current),
                fontsize=7, color=palette.fitted,
                ha="center",
                va="bottom" if peak.current > 0 else "top",
            )
    ax.axhline(0.0, color=palette.border, linewidth=0.6)
    ax.set_xlabel("Potencial (V)")
    ax.set_ylabel("dI/dν (F)" if normalise_by_rate else "Corriente (mA)")
    if len(curves) > 1:
        ax.legend(loc="upper left", frameon=False, fontsize=7, ncol=2)


def plot_gcd(ax, curve: ChargeDischarge, palette: Palette,
             result: Optional[GCDResult] = None) -> None:
    """Potential against time, with the IR drops marked."""
    ax.plot(curve.time, curve.potential, color=palette.data, linewidth=1.1)
    if result is not None:
        clock = float(curve.time[0])
        for branch in result.branches:
            if branch.ir_drop_v > 0.0:
                ax.annotate(
                    "",
                    xy=(clock, branch.v_start),
                    xytext=(clock, branch.v_start + branch.ir_drop_v),
                    arrowprops={"arrowstyle": "<->", "color": palette.danger,
                                "linewidth": 1.0},
                )
                ax.annotate(
                    f"IR {1e3 * branch.ir_drop_v:.0f} mV",
                    xy=(clock, branch.v_start + 0.5 * branch.ir_drop_v),
                    fontsize=7, color=palette.danger, ha="left",
                )
            clock += branch.duration_s
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Potencial (V)")


def plot_nyquist(
    ax, spectrum: Impedance, palette: Palette, result: Optional[EISResult] = None
) -> None:
    """−Z″ against Z′, at equal aspect.

    The aspect matters: a depressed semicircle read off a stretched plot is
    an artefact of the axes, and people do read depression angles off these
    plots by eye.
    """
    ax.plot(
        spectrum.z.real, -spectrum.z.imag, "o", markersize=3.5,
        color=palette.data, label="medido",
    )
    if result is not None and result.fit is not None and result.fit.fitted is not None:
        model = result.fit.fitted
        ax.plot(
            model.real, -model.imag, color=palette.fitted, linewidth=1.2,
            label=f"ajuste: {result.fit.circuit}",
        )
    # A square box with matched spans, rather than equal aspect with free
    # limits: the latter satisfies the aspect by widening the x axis into
    # negative Z', which no impedance ever reaches.
    real = spectrum.z.real
    imaginary = -spectrum.z.imag
    high = float(max(real.max(), imaginary.max())) * 1.05
    low = min(0.0, float(real.min()), float(imaginary.min()))
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Z′ (Ω)")
    ax.set_ylabel("−Z″ (Ω)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)


def plot_bode(
    ax, spectrum: Impedance, palette: Palette, result: Optional[EISResult] = None
) -> None:
    """Modulus and phase against frequency, on twin axes.

    Worth looking at even when the Nyquist plot seems clear: the Nyquist
    plot hides the frequency axis entirely, so a feature spanning three
    decades and one spanning a tenth of one look identical on it.
    """
    ax.loglog(spectrum.frequency, spectrum.modulus, "o", markersize=3.0,
              color=palette.data, label="|Z| medido")
    if result is not None and result.fit is not None and result.fit.fitted is not None:
        ax.loglog(spectrum.frequency, np.abs(result.fit.fitted),
                  color=palette.fitted, linewidth=1.1, label="|Z| ajuste")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("|Z| (Ω)")

    twin = ax.twinx()
    twin.semilogx(spectrum.frequency, -spectrum.phase_deg, "s", markersize=2.5,
                  color=palette.accent, alpha=0.7)
    twin.set_ylabel("−fase (grados)", color=palette.accent)
    twin.tick_params(axis="y", colors=palette.accent, labelsize=8)
    twin.set_ylim(-5.0, 95.0)
    ax.legend(loc="lower left", frameon=False, fontsize=8)


def plot_kk_residuals(ax, result: EISResult, palette: Palette) -> None:
    """The Kramers–Kronig residual against frequency.

    Read the *shape*, not the size. Random scatter about zero means the
    data are consistent; long stretches of one sign mean the cell changed
    while it was being measured, and then no circuit fitted to it means
    anything.
    """
    kk = result.kk
    if kk is None:
        return
    ax.semilogx(result.spectrum.frequency, 100 * kk.residual_real, "o-",
                markersize=2.5, linewidth=0.7, color=palette.data, label="real")
    ax.semilogx(result.spectrum.frequency, 100 * kk.residual_imag, "s-",
                markersize=2.5, linewidth=0.7, color=palette.accent,
                label="imaginaria")
    ax.axhline(0.0, color=palette.border, linewidth=0.6)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Residuo (%)")
    ax.set_title(
        ("consistente" if kk.passes else "NO CONSISTENTE")
        + f" — rachas {kk.runs_ratio:.2f}× el azar",
        fontsize=9,
        color=palette.success if kk.passes else palette.danger,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def plot_rate_capacitance(ax, study: RateStudy, palette: Palette) -> None:
    """Capacitance against scan rate, with the capacitive fraction on top."""
    rates = [1e3 * r for r in study.rates]
    ax.semilogx(rates, [1e3 * c for c in study.capacitances], "o-",
                color=palette.data, linewidth=1.1, label="capacitancia")
    ax.set_xlabel("Velocidad de barrido (mV/s)")
    ax.set_ylabel("Capacitancia (mF)")
    if study.capacitive_fraction:
        twin = ax.twinx()
        ordered = sorted(study.capacitive_fraction.items())
        twin.semilogx(
            [1e3 * r for r, _ in ordered],
            [100 * f for _, f in ordered],
            "s--", color=palette.accent, linewidth=1.0,
        )
        twin.set_ylabel("Fracción capacitiva (%)", color=palette.accent)
        twin.tick_params(axis="y", colors=palette.accent, labelsize=8)
        twin.set_ylim(0.0, 105.0)
    ax.legend(loc="lower left", frameon=False, fontsize=8)


def plot_b_values(ax, study: RateStudy, palette: Palette) -> None:
    """The exponent of ``i = a ν^b`` across the potential window."""
    if not study.b_values:
        return
    potentials = [v.potential for v in study.b_values]
    values = [v.b for v in study.b_values]
    errors = [min(v.b_error, 0.5) for v in study.b_values]
    ax.errorbar(potentials, values, yerr=errors, fmt="o-", color=palette.data,
                linewidth=1.0, markersize=4, capsize=3)
    ax.axhline(1.0, color=palette.success, linewidth=0.8, linestyle="--")
    ax.axhline(0.5, color=palette.warning, linewidth=0.8, linestyle="--")
    ax.annotate("b = 1: superficie", xy=(potentials[0], 1.0), fontsize=7,
                color=palette.success, va="bottom")
    ax.annotate("b = 0.5: difusión", xy=(potentials[0], 0.5), fontsize=7,
                color=palette.warning, va="bottom")
    ax.set_xlabel("Potencial (V)")
    ax.set_ylabel("b")
    ax.set_ylim(0.2, 1.25)


def plot_dunn(ax, analysis, curve, palette: Palette,
              scan_rate: Optional[float] = None) -> None:
    """The capacitive current drawn inside the measured voltammogram.

    This is the figure the Dunn separation exists to produce. The shaded
    area is the surface-controlled part and what is left between it and the
    measured curve is the diffusion-controlled part — so the plot shows
    *where* in the window the diffusive contribution actually sits, which a
    single percentage cannot.
    """
    rate = scan_rate if scan_rate is not None else max(analysis.rates or (0.0,))
    capacitive = analysis.capacitive_current(rate)
    ax.fill_between(analysis.potentials, 0.0, 1e3 * capacitive,
                    color=palette.component_colour(0), alpha=0.35,
                    linewidth=0, label="superficial (k₁ν)")
    if curve is not None and abs(curve.scan_rate - rate) < 1e-9:
        try:
            anodic, _ = curve.sweeps()
            ax.plot(anodic.potential, 1e3 * anodic.current, color=palette.data,
                    linewidth=1.0, label="medido")
        except Exception:                        # noqa: BLE001 - no clean sweep
            pass
    ax.plot(analysis.potentials, 1e3 * (capacitive + analysis.diffusive_current(rate)),
            color=palette.fitted, linewidth=1.0, label="k₁ν + k₂√ν")
    ax.plot(analysis.potentials, 1e3 * analysis.diffusive_current(rate),
            color=palette.accent, linewidth=0.9, linestyle="--",
            label="difusivo (k₂√ν)")
    fraction = analysis.fractions.get(rate)
    # "Superficial", not "capacitivo". A surface-confined redox reaction
    # is surface-controlled, so it sits in k1 by definition, and the word
    # "capacitive" on this plot is what makes people read a correct
    # 99 % as their redox peaks having been swallowed.
    ax.set_title(
        f"{1e3 * rate:g} mV/s" + (f" — {100 * fraction:.0f} % superficial"
                                  if fraction is not None else ""),
        fontsize=8,
    )
    ax.set_xlabel("Potencial (V)")
    ax.set_ylabel("Corriente (mA)")
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)


def plot_tafel(ax, result: CatalysisResult, palette: Palette,
               overpotential=None, current_density=None) -> None:
    """Overpotential against log current density, with the fitted region.

    The fitted stretch is drawn over the data rather than beside it,
    because the useful question about a Tafel fit is not what its slope
    was but *which part of the curve* it used — a slope taken from the
    mass-transport-limited end is a transport property with a kinetic
    name.
    """
    if overpotential is None or current_density is None:
        return
    eta = np.abs(np.asarray(overpotential, dtype=float))
    j = np.abs(np.asarray(current_density, dtype=float))
    usable = j > 1e-8
    ax.semilogx(j[usable], 1e3 * eta[usable], "o", markersize=2.5,
                color=palette.data, label="medido")
    tafel = result.tafel
    if tafel is not None and tafel.valid:
        low, high = tafel.range_mv
        band = (1e3 * eta >= min(low, high)) & (1e3 * eta <= max(low, high)) & usable
        if band.any():
            ax.semilogx(j[band], 1e3 * eta[band], color=palette.fitted,
                        linewidth=2.0, alpha=0.8,
                        label=f"{tafel.slope_mv_per_decade:.0f} mV/dec")
    if result.overpotential_at_benchmark is not None:
        ax.axvline(result.benchmark_ma_cm2, color=palette.accent,
                   linewidth=0.8, linestyle=":")
        ax.annotate(
            f"η = {1e3 * result.overpotential_at_benchmark:.0f} mV",
            xy=(result.benchmark_ma_cm2, 1e3 * result.overpotential_at_benchmark),
            fontsize=8, color=palette.accent, ha="left", va="top",
        )
    ax.set_xlabel("|j| (mA/cm²)")
    ax.set_ylabel("Sobrepotencial (mV)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)


def plot_ragone(ax, points: Sequence[tuple[float, float, str]],
                palette: Palette) -> None:
    """Energy against power, both logarithmic.

    ``points`` are ``(Wh/kg, W/kg, label)``. Logarithmic on both axes
    because devices differ by orders of magnitude on both, and because a
    linear Ragone plot compresses every supercapacitor into the corner.
    """
    for index, (energy, power, label) in enumerate(points):
        if energy <= 0 or power <= 0:
            continue
        ax.loglog([power], [energy], "o", markersize=7,
                  color=palette.component_colour(index), label=label)
    ax.set_xlabel("Potencia específica (W/kg)")
    ax.set_ylabel("Energía específica (Wh/kg)")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.2)


def plot_cycling(ax, cycles: Sequence[int], capacity: Sequence[float],
                 efficiency: Sequence[float], palette: Palette) -> None:
    """Capacity retention and coulombic efficiency against cycle number."""
    ax.plot(cycles, [1e3 * c for c in capacity], "o-", color=palette.data,
            markersize=3, linewidth=1.0, label="capacidad")
    ax.set_xlabel("Ciclo")
    ax.set_ylabel("Capacidad (mC)")
    if efficiency:
        twin = ax.twinx()
        twin.plot(cycles[: len(efficiency)], [100 * e for e in efficiency], "s",
                  color=palette.accent, markersize=3)
        twin.set_ylabel("Eficiencia culómbica (%)", color=palette.accent)
        twin.tick_params(axis="y", colors=palette.accent, labelsize=8)
        twin.set_ylim(0.0, 105.0)
    ax.legend(loc="lower left", frameon=False, fontsize=8)


def plot_drt(ax, result, palette: Palette) -> None:
    """γ(τ) against τ, logarithmic in τ, with the resolved processes marked.

    Logarithmic because the whole point of a DRT is separating time
    constants that span decades, and the peak areas — not the peak
    heights — are the resistances, so the shaded basins are the number to
    read. The regularisation is in the title because a DRT without it is
    not a measurement: it is one choice of how much structure to believe,
    and another choice gives a defensible distribution with a different
    number of peaks.
    """
    if result is None:
        placeholder(ax, "Carga un espectro de impedancia", palette)
        return
    tau = np.asarray(result.tau_s, dtype=float)
    gamma = np.asarray(result.gamma, dtype=float)
    ax.semilogx(tau, gamma, color=palette.data, linewidth=1.4)
    ax.fill_between(tau, 0.0, gamma, color=palette.data, alpha=0.15)

    # The y-axis is scaled to the MEASURED range of time constants and
    # the rest is shaded. Outside it gamma is not determined by the data
    # and does not sit quietly at zero either: the unmeasured slow end
    # collects whatever the diffusion tail implies, which on the demo
    # spectrum is a spike seven hundred times the real peaks. Autoscaled
    # to that, the plot is a flat line with a wall at one end and every
    # resolved process is invisible -- the same failure the peak finder
    # already guards against, arriving through the figure instead.
    low, high = result.measured_s
    inside = ((tau >= low) & (tau <= high)) if high > low else np.ones(
        tau.shape, dtype=bool)
    if inside.any() and gamma[inside].max() > 0:
        ax.set_ylim(0.0, 1.35 * float(gamma[inside].max()))
    if high > low:
        for start, stop in ((tau.min(), low), (high, tau.max())):
            if stop > start:
                ax.axvspan(start, stop, color=palette.text_muted,
                           alpha=0.18, linewidth=0)
        ax.annotate("sin medir", xy=(high, 0.0), xytext=(6, 6),
                    textcoords="offset points", fontsize=7,
                    color=palette.text_muted)

    top = ax.get_ylim()[1]
    for index, (peak, resistance) in enumerate(
        zip(result.peaks_s, result.peak_resistances)
    ):
        colour = palette.component_colour(index)
        ax.axvline(peak, color=colour, linestyle="--", linewidth=0.9, alpha=0.8)
        ax.annotate(f"{resistance:.3g} Ω", xy=(peak, top),
                    xytext=(0, -10 - 11 * (index % 3)),
                    textcoords="offset points", fontsize=7, color=colour,
                    ha="center")
    ax.set_xlabel("τ (s)")
    ax.set_ylabel("γ(τ) (Ω)")
    ax.set_title(f"λ = {result.regularisation:.3g}   residuo "
                 f"{100 * result.residual:.2f} %", fontsize=9)
    ax.grid(True, which="both", alpha=0.2)


def plot_complex_capacitance(ax_real, ax_imag, result, palette: Palette) -> None:
    """C′(ω) and C″(ω), with τ₀ marked on the second.

    Straight from the data: ``C(ω) = 1/(jωZ)`` needs no circuit and no
    mass, and the maximum of C″ is the device's relaxation time — the
    boundary between the frequencies where it behaves as a capacitor and
    the ones where it behaves as a resistor.
    """
    if result is None:
        placeholder(ax_real, "Carga un espectro de impedancia", palette)
        placeholder(ax_imag, "", palette)
        return
    frequency = np.asarray(result.frequency, dtype=float)
    ax_real.semilogx(frequency, 1e3 * np.asarray(result.real), color=palette.data,
                     linewidth=1.4, label="C′")
    ax_real.semilogx(frequency, 1e3 * np.asarray(result.series),
                     color=palette.accent, linewidth=1.0, linestyle="--",
                     label="C serie")
    ax_real.set_xlabel("Frecuencia (Hz)")
    ax_real.set_ylabel("C′ (mF)")
    ax_real.legend(loc="best", frameon=False, fontsize=8)
    ax_real.grid(True, which="both", alpha=0.2)

    ax_imag.semilogx(frequency, 1e3 * np.asarray(result.imaginary),
                     color=palette.fitted, linewidth=1.4)
    if result.relaxation_frequency_hz:
        ax_imag.axvline(result.relaxation_frequency_hz, color=palette.accent,
                        linestyle="--", linewidth=0.9)
        ax_imag.annotate(f"τ₀ = {result.relaxation_s:.3g} s",
                         xy=(result.relaxation_frequency_hz, 0.0),
                         xytext=(4, 12), textcoords="offset points",
                         fontsize=8, color=palette.accent)
    ax_imag.set_xlabel("Frecuencia (Hz)")
    ax_imag.set_ylabel("C″ (mF)")
    ax_imag.grid(True, which="both", alpha=0.2)


def plot_capacitance_comparison(ax, comparison, palette: Palette) -> None:
    """The same electrode measured three ways, side by side.

    The disagreement IS the result. What a device delivers is the GCD
    value; the EIS one is measured with 10 mV about a fixed point where
    nothing is rate-limited, and is an upper bound the device never sees.
    A spread above 30 % is worth more than any one of the three numbers,
    so it is written on the plot rather than left to be worked out.
    """
    if comparison is None or not comparison.entries:
        placeholder(ax, "Hacen falta al menos dos métodos", palette)
        return
    entries = list(comparison.entries)
    labels = [entry.method for entry in entries]
    # F/g where the mass is known, farads otherwise: mixing the two on
    # one axis would put a 2 F/g and a 2 F bar at the same height.
    per_gram = all(entry.per_gram is not None for entry in entries)
    values = [entry.per_gram if per_gram else 1e3 * entry.farads
              for entry in entries]
    positions = np.arange(len(entries))
    ax.bar(positions, values,
           color=[palette.component_colour(i) for i in range(len(entries))],
           width=0.6)
    for position, entry, value in zip(positions, entries, values):
        ax.annotate(entry.condition, xy=(position, value), xytext=(0, 4),
                    textcoords="offset points", fontsize=7, ha="center",
                    color=palette.text)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("C (F/g)" if per_gram else "C (mF)")
    if comparison.spread is not None:
        ax.set_title(f"dispersión entre métodos: "
                     f"{100 * comparison.spread:.0f} %", fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)


__all__ = [
    "plot_b_values",
    "plot_bode",
    "plot_cv",
    "plot_capacitance_comparison",
    "plot_complex_capacitance",
    "plot_cycling",
    "plot_drt",
    "plot_dunn",
    "plot_gcd",
    "plot_kk_residuals",
    "plot_nyquist",
    "plot_ragone",
    "plot_rate_capacitance",
    "plot_tafel",
]
