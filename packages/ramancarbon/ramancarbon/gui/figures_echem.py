"""Composite, multi-panel electrochemistry figures.

Written against matplotlib axes and never against Tk, like
``plots_echem``. The difference is what it is for: ``plots_echem`` draws
the one figure a tab shows, and this assembles the figure that goes in a
manuscript — several panels, chosen and laid out by the user, with their
letters, exported at a stated width and resolution.

**Why it exists.** The multi-panel figures of this project's analyses
were written by hand, one script per analysis, and that is a dead end in
two directions at once: the person who wants one has to be able to write
matplotlib, and the panels drift away from what the window shows because
nothing keeps them together. Every panel here draws from the same
``EchemSession`` the tabs draw from.

**A figure is a VALUE**, the same rule the package's plotting engine
already follows: :class:`CompositeFigure` is the choice of panels, the
layout and the options, it converts to a dictionary and back, and
:func:`draw_composite` is the only thing that touches matplotlib. So a
figure survives a saved project and comes back identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .theme import Palette

#: Letters for the panels, in order. Lower case with brackets, which is
#: what almost every journal asks for and what the eye reads as a panel
#: label rather than as part of the data.
LETTERS = "abcdefghijkl"

#: Default colours for a series of samples or of scan rates, in order.
#: Okabe–Ito, like the rest of the package: safe for colour blindness and
#: still separable in greyscale. NOT matplotlib's default cycle.
SERIES_COLOURS = ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00",
                  "#56B4E9", "#F0E442", "#000000")


def _colour(index: int) -> str:
    return SERIES_COLOURS[index % len(SERIES_COLOURS)]


@dataclass
class CompositeFigure:
    """Which panels, laid out how, with which options.

    Everything here is a CHOICE and none of it is data, which is why it
    is a separate object: the same measurements make a two-panel figure
    at one column width and a six-panel one at two, and neither is more
    correct than the other.
    """

    panels: list[str] = field(default_factory=list)
    columns: int = 2
    width_in: float = 7.2
    """Total figure width in inches. One journal column is about 3.3 and
    two are about 7, and the pairing of width with font size is what
    makes a figure legible — drawing at 7 and shrinking to 3.3 halves
    every label."""
    panel_height_in: float = 3.1
    dpi: int = 220
    letters: bool = True
    legends: bool = True
    grid: bool = False
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "panels": list(self.panels), "columns": self.columns,
            "width_in": self.width_in, "panel_height_in": self.panel_height_in,
            "dpi": self.dpi, "letters": self.letters, "legends": self.legends,
            "grid": self.grid, "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompositeFigure":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def rows(self) -> int:
        return max(1, math.ceil(len(self.panels) / max(1, self.columns)))

    def figsize(self) -> tuple[float, float]:
        return (self.width_in, self.panel_height_in * self.rows)


@dataclass(frozen=True)
class PanelSpec:
    """One panel the composite figure can hold."""

    key: str
    label: str
    """What the checkbox says."""
    needs: str
    """What has to be loaded: ``"rates"``, ``"eis_series"``, ``"eis"``,
    ``"cv"``. A panel whose data are missing is not offered, rather than
    offered and then drawn empty."""
    draw: Callable[..., None]
    note: str = ""
    """One line on what the panel is for, shown next to the checkbox."""


# -- what each panel needs from the session ----------------------------


def _rate_curves(session) -> list:
    return sorted(session.rate_series, key=lambda c: c.scan_rate)


def _eis_list(session) -> list:
    if session.eis_series:
        return list(session.eis_series)
    return [session.eis] if session.eis is not None else []


def _tidy(axis, spec: CompositeFigure) -> None:
    axis.tick_params(which="both", direction="in", top=True, right=True,
                     labelsize=8)
    if spec.grid:
        axis.grid(True, which="major", linewidth=0.4, alpha=0.35)


# -- the voltammetry panels --------------------------------------------


def _rate_label(curve) -> str:
    """The scan rate as a person would write it.

    Three significant figures, because a rate MEASURED off |dE/dt| comes
    back as 4.99952 mV/s and a legend that says so is reporting the
    clock's precision as if it were a setting.
    """
    return f"{1e3 * curve.scan_rate:.3g} mV/s"


def _panel_cv_series(axis, session, spec, palette) -> None:
    curves = _rate_curves(session)
    label = "E / V"
    unit = "i / A"
    for index, curve in enumerate(curves):
        # The REPORTING cycle's own arrays. Masking the whole file by
        # which potentials appear in the cycle looks equivalent and is
        # not: `isin` matches VALUES, and a voltammogram sweeps the same
        # potentials once per cycle, so every cycle in the file came
        # through and the panel drew the lot.
        cycle, _ = curve.reporting_cycle()
        values, label, _ = cycle.electrode.on_scale(
            cycle.potential, session.potential_scale_choice)
        current = cycle.electrode.specific(cycle.current, "mass")
        if current is None:
            current, unit = cycle.current, "i / A"
        else:
            unit = "i / A g$^{-1}$"
        axis.plot(values, current, color=_colour(index), linewidth=1.3,
                  label=_rate_label(curve))
    axis.set_ylabel(unit)
    axis.axhline(0.0, color=palette.border, linewidth=0.6)
    axis.set_xlabel(label)
    if spec.legends and curves:
        axis.legend(frameon=False, fontsize=7.5, ncol=2, handlelength=1.4,
                    columnspacing=1.0)
    _tidy(axis, spec)


def _panel_cv_capacitance(axis, session, spec, palette) -> None:
    from ..echem.cv import capacitance, coulombic_efficiency

    from ..echem.cv import reversible_capacitance

    curves = _rate_curves(session)
    # `Capacitance` carries its own specific values, by convention and
    # by basis; it is NOT a bare number, and dividing one by a mass is
    # how this panel first came back with a type error instead of a
    # figure.
    per_gram = bool(curves and curves[0].electrode.mass_mg)
    rates, loop, reversible, efficiency = [], [], [], []
    for curve in curves:
        cycle, _ = curve.reporting_cycle()
        value, _ = capacitance(cycle)
        other = reversible_capacitance(cycle)
        rates.append(1e3 * curve.scan_rate)
        loop.append(value.specific_f_per_g if per_gram else value.farads)
        reversible.append(other.specific_f_per_g if per_gram
                          else other.farads)
        efficiency.append(coulombic_efficiency(cycle))
    axis.plot(rates, loop, "o-", color=_colour(0), markersize=5,
              linewidth=1.3, label="lazo ÷ 2νΔV")
    axis.plot(rates, reversible, "s-", color=_colour(1), markersize=5,
              linewidth=1.3, label="carga de reducción ÷ ΔV")
    axis.set_xlabel("ν / mV s$^{-1}$")
    axis.set_ylabel("C / F g$^{-1}$" if per_gram else "C / F")
    twin = axis.twinx()
    twin.plot(rates, efficiency, "^--", color=_colour(2), markersize=5,
              linewidth=1.1)
    twin.set_ylabel("Q$_{red}$/Q$_{ox}$", color=_colour(2), fontsize=8.5)
    twin.tick_params(axis="y", colors=_colour(2), labelsize=8, direction="in")
    if spec.legends:
        axis.legend(frameon=False, fontsize=7.5, handlelength=1.5)
    axis.tick_params(which="both", direction="in", top=True, labelsize=8)


def _panel_cv_balance(axis, session, spec, palette) -> None:
    """ΔQ against cycle duration: the parasitic current, measured.

    The slope IS the leak averaged over the cycle, and the intercept has
    to be zero — that is the check on the model, and it is why this panel
    belongs next to any capacitance quoted from a loop.
    """
    from ..echem.cv import elapsed_time, integrate_charge

    curves = _rate_curves(session)
    durations, deltas = [], []
    for index, curve in enumerate(curves):
        cycle, _ = curve.reporting_cycle()
        time, _ = elapsed_time(cycle)
        oxidation, absolute, _ = integrate_charge(cycle)
        reduction = absolute - oxidation
        durations.append(float(time[-1]))
        deltas.append(1e6 * (oxidation - reduction))
        axis.plot([durations[-1]], [deltas[-1]], "o", color=_colour(index),
                  markersize=6, zorder=3, label=_rate_label(curve))
    leak = getattr(session.result, "rates", None)
    leak = getattr(leak, "leakage", None) if leak else None
    if leak is not None and durations:
        grid = np.linspace(0.0, max(durations) * 1.05, 32)
        axis.plot(grid, 1e6 * (leak.leakage_a * grid + leak.offset_c),
                  "-", color=palette.fitted, linewidth=1.2, zorder=2,
                  label=f"⟨i⟩ = {1e6 * leak.leakage_a:.2f} µA  "
                        f"(R² = {leak.r_squared:.3f})")
    axis.axhline(0.0, color=palette.border, linewidth=0.6)
    axis.set_xlabel("duración del ciclo / s")
    axis.set_ylabel("ΔQ = Q$_{ox}$ − Q$_{red}$  /  µC")
    if spec.legends:
        axis.legend(frameon=False, fontsize=7, handlelength=1.3)
    _tidy(axis, spec)


def _panel_cv_terms(axis, session, spec, palette) -> None:
    """The three contributions to i(ν), potential by potential."""
    study = getattr(session.result, "rates", None)
    models = getattr(study, "full_model", None) if study else None
    if not models:
        return
    slowest = min(session.rate_series, key=lambda c: c.scan_rate).scan_rate
    potentials = [m.potential for m in models]
    capacitive = [1e6 * abs(m.capacitance_f * slowest) for m in models]
    diffusive = [1e6 * abs(m.diffusive * math.sqrt(slowest)) for m in models]
    parasitic = [1e6 * abs(m.leakage_a) for m in models]
    # Cumulative boundaries drawn as LINES with the bands filled between
    # them, rather than `stackplot`. Visually the same stack; the
    # difference is that a stackplot leaves no Line2D behind, so the
    # panel came out of "save the figure's numbers" with nothing but its
    # header — a figure whose numbers cannot be exported has to be
    # retyped to be used anywhere else.
    first = np.asarray(capacitive)
    second = first + np.asarray(diffusive)
    third = second + np.asarray(parasitic)
    base = np.zeros_like(first)
    for upper, lower, colour, label in (
        (first, base, _colour(0), "k$_{cap}$·ν"),
        (second, first, _colour(1), "k$_{dif}$·√ν"),
        (third, second, _colour(2), "i$_{par}$"),
    ):
        axis.fill_between(potentials, lower, upper, color=colour, alpha=0.8,
                          linewidth=0.0)
        axis.plot(potentials, upper, "-", color=colour, linewidth=1.1,
                  label=label)
    axis.set_xlabel("E / V")
    axis.set_ylabel(f"|i| a {1e3 * slowest:.3g} mV/s  /  µA")
    if spec.legends:
        axis.legend(frameon=False, fontsize=7.5, loc="upper left")
    _tidy(axis, spec)


def _panel_dunn(axis, session, spec, palette) -> None:
    from .plots_echem import plot_dunn

    result = session.result
    analysis = getattr(result, "dunn", None) if result else None
    curve = session.cv or (session.rate_series[0]
                           if session.rate_series else None)
    if analysis is None or curve is None:
        return
    plot_dunn(axis, analysis, curve, palette)
    _tidy(axis, spec)


# -- the impedance panels ----------------------------------------------


def _panel_nyquist(axis, session, spec, palette) -> None:
    """The whole spectrum, log–log, with the fit.

    Logarithmic on both axes because a porous electrode spans four
    decades of |Z|: on linear axes the four lowest-frequency points fill
    the figure and everything else is a dot at the origin.
    """
    analyses = session.eis_analyses()
    for index, (spectrum, analysis) in enumerate(zip(_eis_list(session),
                                                     analyses)):
        axis.plot(spectrum.z.real, -spectrum.z.imag, "o",
                  color=_colour(index), markersize=3.2,
                  markerfacecolor="none", markeredgewidth=0.9, zorder=3,
                  label=spectrum.name)
        fit = analysis.fit if analysis else None
        if fit is not None and fit.fitted is not None:
            axis.plot(fit.fitted.real, -fit.fitted.imag, "-",
                      color=_colour(index), linewidth=1.2, zorder=2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Z′ / Ω")
    axis.set_ylabel("−Z″ / Ω")
    if spec.legends:
        axis.legend(frameon=False, fontsize=7.5, handlelength=1.2)
    axis.tick_params(which="both", direction="in", top=True, right=True,
                     labelsize=8)


def _panel_nyquist_corner(axis, session, spec, palette) -> None:
    """The high-frequency corner, linear and SQUARE.

    A Nyquist plot is read by its shape, so an aspect ratio other than
    one turns a 45° line into whatever angle the box happens to give.
    """
    spectra = _eis_list(session)
    analyses = session.eis_analyses()
    if not spectra:
        return
    limit = 6.0 * max(float(s.z.real.min()) for s in spectra)
    for index, (spectrum, analysis) in enumerate(zip(spectra, analyses)):
        keep = -spectrum.z.imag <= limit
        axis.plot(spectrum.z.real[keep], -spectrum.z.imag[keep], "o",
                  color=_colour(index), markersize=3.6,
                  markerfacecolor="none", markeredgewidth=0.9, zorder=3,
                  label=spectrum.name)
        fit = analysis.fit if analysis else None
        if fit is not None and fit.fitted is not None:
            axis.plot(fit.fitted.real[keep], -fit.fitted.imag[keep], "-",
                      color=_colour(index), linewidth=1.2, zorder=2)
        if analysis is not None and analysis.series_resistance is not None:
            axis.plot([analysis.series_resistance], [0.0], "v",
                      color=_colour(index), markersize=5, zorder=4)
    axis.plot([0.0, limit], [0.0, limit], linestyle=(0, (4, 2)),
              color=palette.border, linewidth=0.9, zorder=1)
    axis.set_xlim(0.0, limit)
    axis.set_ylim(0.0, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Z′ / Ω")
    axis.set_ylabel("−Z″ / Ω")
    if spec.legends:
        axis.legend(frameon=False, fontsize=7, handlelength=1.2)
    axis.tick_params(which="both", direction="in", top=True, right=True,
                     labelsize=8)


def _panel_bode(axis, session, spec, palette) -> None:
    """|Z| and the phase, with each spectrum's knee marked where it was
    measured — the crossing of −45°, not the maximum of −Z″."""
    from ..echem.eis import KNEE_PHASE_DEG

    phase = axis.twinx()
    for index, (spectrum, analysis) in enumerate(
            zip(_eis_list(session), session.eis_analyses())):
        axis.plot(spectrum.frequency, spectrum.modulus, "-",
                  color=_colour(index), linewidth=1.3, zorder=3,
                  label=spectrum.name)
        phase.plot(spectrum.frequency, spectrum.phase_deg,
                   linestyle=(0, (3, 2)), color=_colour(index),
                   linewidth=1.1, zorder=2)
        knee = analysis.knee_frequency if analysis else None
        if knee:
            phase.plot([knee], [KNEE_PHASE_DEG], "o", color=_colour(index),
                       markersize=5, markeredgecolor=palette.surface,
                       markeredgewidth=0.8, zorder=5)
    phase.axhline(KNEE_PHASE_DEG, color=palette.border, linewidth=0.7,
                  linestyle=(0, (1, 2)), zorder=1)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("f / Hz")
    axis.set_ylabel("|Z| / Ω  (continua)")
    phase.set_ylabel("fase / grados  (discontinua)", fontsize=8.5)
    phase.set_ylim(-95.0, 5.0)
    phase.tick_params(which="both", direction="in", labelsize=8)
    axis.tick_params(which="both", direction="in", top=True, labelsize=8)
    if spec.legends:
        axis.legend(frameon=False, fontsize=7.5, handlelength=1.2,
                    loc="lower left")


def _panel_complex_capacitance(axis, session, spec, palette) -> None:
    """C′(f) and C″(f): the capacitance with no circuit anywhere near it."""
    from ..echem.capacitance import complex_capacitance

    second = axis.twinx()
    for index, spectrum in enumerate(_eis_list(session)):
        analysis = complex_capacitance(spectrum)
        axis.plot(analysis.frequency, 1e6 * analysis.real, "-",
                  color=_colour(index), linewidth=1.4, zorder=3,
                  label=spectrum.name)
        second.plot(analysis.frequency, 1e6 * analysis.imaginary,
                    linestyle=(0, (3, 2)), color=_colour(index),
                    linewidth=1.0, zorder=2)
        if analysis.relaxation_frequency_hz:
            peak = float(np.interp(analysis.relaxation_frequency_hz,
                                   analysis.frequency,
                                   1e6 * analysis.imaginary))
            second.plot([analysis.relaxation_frequency_hz], [peak], "s",
                        color=_colour(index), markersize=4.5,
                        markeredgecolor=palette.surface, markeredgewidth=0.8,
                        zorder=5)
    axis.set_xscale("log")
    axis.set_xlabel("f / Hz")
    axis.set_ylabel("C′ / µF  (continua)")
    second.set_ylabel("C″ / µF  (discontinua)", fontsize=8.5)
    second.tick_params(which="both", direction="in", labelsize=8)
    axis.tick_params(which="both", direction="in", top=True, labelsize=8)
    if spec.legends:
        axis.legend(frameon=False, fontsize=7.5, handlelength=1.2)


def _panel_drt_series(axis, session, spec, palette) -> None:
    """γ(τ) for every spectrum, each normalised to its own maximum.

    Normalised and linear on purpose. Several samples' γ differ by an
    order of magnitude, so on a shared linear axis the tallest flattens
    the rest into the baseline; and on a logarithmic one the sparseness
    of the non-negative solution draws as a comb of spikes that says
    nothing. What the panel is for is WHERE the weight sits.
    """
    from ..echem.eis import drt

    for index, spectrum in enumerate(_eis_list(session)):
        result = drt(spectrum, regularisation=session.drt_regularisation)
        low, high = result.measured_s
        inside = (result.tau_s >= low) & (result.tau_s <= high)
        gamma = result.gamma[inside]
        peak = float(gamma.max()) if gamma.size else 0.0
        if peak <= 0.0:
            continue
        axis.plot(result.tau_s[inside], gamma / peak, "-",
                  color=_colour(index), linewidth=1.3, zorder=3,
                  label=f"{spectrum.name}  (γ$_{{máx}}$ = {peak:.0f} Ω)")
        for tau in result.peaks_s:
            axis.plot([tau], [0.0], "^", color=_colour(index), markersize=5,
                      clip_on=False, zorder=5)
    axis.set_xscale("log")
    axis.set_ylim(0.0, 1.18)
    axis.set_xlabel("τ / s")
    axis.set_ylabel("γ(τ) / γ$_{máx}$")
    if spec.legends:
        axis.legend(frameon=False, fontsize=7, handlelength=1.2)
    _tidy(axis, spec)


def _panel_eis_residual(axis, session, spec, palette) -> None:
    """Where the circuit fails, frequency by frequency.

    The impedance equivalent of a Rietveld difference curve: a pretty
    mean residual with a systematic wobble in it is a worse fit than an
    ugly one with no structure.
    """
    for index, (spectrum, analysis) in enumerate(
            zip(_eis_list(session), session.eis_analyses())):
        fit = analysis.fit if analysis else None
        if fit is None or fit.fitted is None:
            continue
        residual = 100.0 * np.abs(spectrum.z - fit.fitted) / np.abs(spectrum.z)
        axis.plot(spectrum.frequency, residual, "o-", color=_colour(index),
                  markersize=2.8, linewidth=1.0, zorder=3,
                  label=f"{spectrum.name}: {fit.circuit} "
                        f"({fit.residual_pct:.2f} %)")
    axis.axhline(0.0, color=palette.border, linewidth=0.6)
    axis.set_xscale("log")
    axis.set_xlabel("f / Hz")
    axis.set_ylabel("|Z − Z$_{aj}$| / |Z|  /  %")
    if spec.legends:
        axis.legend(frameon=False, fontsize=7, handlelength=1.3)
    _tidy(axis, spec)


def _panel_kk(axis, session, spec, palette) -> None:
    from .plots_echem import plot_kk_residuals

    result = getattr(session.result, "eis", None) if session.result else None
    if result is not None:
        plot_kk_residuals(axis, result, palette)


#: Every panel the composite figure can hold, in the order they are
#: offered. The order is the order a reader goes through them: the
#: measurement first, then what was derived from it, then what checks it.
ECHEM_PANELS: tuple[PanelSpec, ...] = (
    PanelSpec("cv_serie", "CV: la serie de velocidades", "rates",
              _panel_cv_series,
              "los voltamperogramas del ciclo elegido, uno por velocidad"),
    PanelSpec("cv_capacitancia", "CV: capacitancia y eficiencia contra ν",
              "rates", _panel_cv_capacitance,
              "los dos convenios de capacitancia y Q_red/Q_ox encima"),
    PanelSpec("cv_balance", "CV: balance de carga contra la duración",
              "rates", _panel_cv_balance,
              "ΔQ contra t: la pendiente es la corriente parásita y la "
              "ordenada tiene que ser cero"),
    PanelSpec("cv_terminos", "CV: las tres contribuciones a i(ν)", "rates",
              _panel_cv_terms,
              "k_cap·ν, k_dif·√ν e i_par potencial a potencial; necesita "
              "cuatro velocidades o más"),
    PanelSpec("cv_dunn", "CV: la separación de Dunn", "cv", _panel_dunn,
              "el ciclo cerrado con la parte superficial sembrada dentro"),
    PanelSpec("eis_nyquist", "EIS: Nyquist completo (log-log)", "eis",
              _panel_nyquist,
              "todo el barrido; en lineal las cuatro frecuencias más bajas "
              "se comen la figura"),
    PanelSpec("eis_rincon", "EIS: el rincón de alta frecuencia (cuadrado)",
              "eis", _panel_nyquist_corner,
              "lineal y a escala uno a uno, que es como se lee la forma; "
              "con la R_s medida marcada"),
    PanelSpec("eis_bode", "EIS: Bode con el codo marcado", "eis",
              _panel_bode,
              "|Z| y la fase, con el cruce de −45° de cada espectro"),
    PanelSpec("eis_capacitancia", "EIS: C′(f) y C″(f)", "eis",
              _panel_complex_capacitance,
              "la capacitancia sacada de Z sin ajustar nada, y τ₀ en el "
              "máximo de C″"),
    PanelSpec("eis_drt", "EIS: la DRT normalizada", "eis",
              _panel_drt_series,
              "γ(τ) de cada espectro sobre su propio máximo, solo en el "
              "intervalo medido"),
    PanelSpec("eis_residuo", "EIS: el residuo del ajuste", "eis",
              _panel_eis_residual,
              "dónde falla el circuito, frecuencia a frecuencia"),
    PanelSpec("eis_kk", "EIS: el residuo de Kramers-Kronig", "eis",
              _panel_kk,
              "se lee la FORMA y no el tamaño: rachas de un signo son "
              "deriva de la celda"),
)

PANELS_BY_KEY: dict[str, PanelSpec] = {p.key: p for p in ECHEM_PANELS}


def available_panels(session) -> list[PanelSpec]:
    """The panels whose data are loaded.

    A panel without data is left OUT of the list rather than offered and
    then drawn empty: an empty panel in a six-panel figure reads as a
    bug, and there is no way to tell it from one whose data were there
    and said nothing.
    """
    have = {
        "rates": len(session.rate_series) >= 2,
        "cv": session.cv is not None or bool(session.rate_series),
        "eis": session.eis is not None or bool(session.eis_series),
    }
    return [panel for panel in ECHEM_PANELS if have.get(panel.needs, False)]


def draw_composite(figure, spec: CompositeFigure, session,
                   palette: Palette) -> list[str]:
    """Draw ``spec`` onto a matplotlib figure. Returns what went wrong.

    A panel that raises is reported and the rest of the figure is still
    drawn — building a six-panel figure and getting a traceback instead
    of the five panels that worked is the wrong trade.
    """
    figure.clear()
    keys = [k for k in spec.panels if k in PANELS_BY_KEY]
    if not keys:
        from .base import placeholder

        placeholder(figure.add_subplot(111), palette,
                    "Elige al menos un panel")
        return []
    columns = max(1, min(spec.columns, len(keys)))
    rows = math.ceil(len(keys) / columns)
    problems: list[str] = []
    for index, key in enumerate(keys):
        panel = PANELS_BY_KEY[key]
        axis = figure.add_subplot(rows, columns, index + 1)
        try:
            panel.draw(axis, session, spec, palette)
        except Exception as error:          # noqa: BLE001 - reported, not raised
            problems.append(f"{panel.label}: {error}")
            axis.set_axis_off()
            axis.text(0.5, 0.5, f"{panel.label}\nno se pudo dibujar",
                      transform=axis.transAxes, ha="center", va="center",
                      fontsize=8, color=palette.danger)
            continue
        if spec.letters:
            # ABOVE the frame, not inside it. Inside, the letter lands on
            # whatever corner the data or the legend happen to want, and
            # there is no corner that is free in every panel; outside, it
            # never collides and it is what most journals ask for anyway.
            axis.text(-0.02, 1.03, f"({LETTERS[index % len(LETTERS)]})",
                      transform=axis.transAxes, ha="left", va="bottom",
                      fontsize=9, fontweight="bold")
    if spec.title:
        figure.suptitle(spec.title, fontsize=10)
    figure.tight_layout(h_pad=1.6, w_pad=2.2,
                        rect=(0, 0, 1, 0.96 if spec.title else 1))
    return problems


def save_composite(spec: CompositeFigure, session, palette: Palette,
                   path) -> "Any":
    """Render ``spec`` to a file at its own width and resolution.

    Drawn into a figure of its own rather than saved from the one on
    screen: the canvas is whatever size the pane happens to be, and a
    figure drawn at one width and saved at another arrives with its
    labels at the wrong size.
    """
    from pathlib import Path

    import matplotlib
    from matplotlib.figure import Figure

    from .theme import matplotlib_style

    target = Path(path)
    with matplotlib.rc_context(matplotlib_style(palette)):
        figure = Figure(figsize=spec.figsize(), dpi=spec.dpi)
        problems = draw_composite(figure, spec, session, palette)
        figure.savefig(target, dpi=spec.dpi, bbox_inches="tight")
    return target, problems


__all__ = [
    "CompositeFigure",
    "ECHEM_PANELS",
    "LETTERS",
    "PANELS_BY_KEY",
    "PanelSpec",
    "SERIES_COLOURS",
    "available_panels",
    "draw_composite",
    "save_composite",
]
