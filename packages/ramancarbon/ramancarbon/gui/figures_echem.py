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

**The axis labels default to ENGLISH and the interface stays Spanish.**
Those are two different audiences: the window is for the person running
the experiment and the figure is for whoever reads the paper, and this
group publishes in English. The language is an option, not a constant,
because the same measurement goes into a thesis chapter and into a paper
and they are not always in the same language.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .theme import Palette

#: Letters for the panels, in order. Lower case in brackets, bold, which
#: is what almost every journal asks for and what the eye reads as a
#: panel label rather than as part of the data.
LETTERS = "abcdefghijkl"

#: Axis labels and legend entries, in both languages.
#:
#: A table rather than inline strings because a figure that mixes "E / V"
#: with "duración del ciclo" is the single most visible way a figure
#: looks unfinished, and inline strings guarantee it as soon as anyone
#: adds a panel.
LABELS: dict[str, dict[str, str]] = {
    "en": {
        'time_s': 't / s',
        'gcd_potential': 'E / V vs {reference}',
        'current_density_ma': 'i / mA',
        'current_density_a_g': 'i / A g$^{-1}$',
        'capacitance_uf': 'C / µF',
        'cycle': 'cycle number',
        'retention': 'capacitance retention / %',
        'coulombic': 'coulombic efficiency / %',
        'energy_wh': 'E / Wh kg$^{-1}$',
        'power_w': 'P / W kg$^{-1}$',
        'ir_drop_mv': 'IR drop / mV',
        'esr_ohm': 'ESR / $\\Omega$',
        'leakage_ua': 'parasitic current / µA',
        'capacity_mah': 'Q / mAh g$^{-1}$',
        'overpotential': '$\\eta$ / V',
        'log_j': 'log(|j| / mA cm$^{-2}$)',
        'j_ma_cm2': '|j| / mA cm$^{-2}$',
        'tafel_fit': '{slope} mV dec$^{-1}$',
        'tafel_region': 'fitted region',
        'asymmetric_note': 'charge and discharge at different currents',
        'interpolated': 'interpolated from the symmetric points',
        'measured_point': 'measured',
        'ceiling': '$\\Delta V^2/4R$ ceiling',
        'fade_fit': '{model}, R$^2$ = {r2}',
        'benchmark': '{j} mA cm$^{-2}$',
        "potential": "E / V vs {reference}",
        "inverse_c2": "C$^{-2}$ / F$^{-2}$",
        "ms_fit": "{f} Hz: E$_{fb}$ = {efb} V, N = {n} cm$^{-3}$",
        "ms_unused": "outside the linear region",
        "ms_flatband": "E$_{fb}$",
        "potential_plain": "E / V",
        "rhe": "E / V vs RHE",
        "current_mass": "i / A g$^{-1}$",
        "current": "i / A",
        "rate": r"$\nu$ / mV s$^{-1}$",
        "capacitance_mass": "C / F g$^{-1}$",
        "capacitance": "C / F",
        "loop": r"loop $\div\ 2\nu\Delta V$  (leak included)",
        "reduction": r"reduction charge $\div\ \Delta V$",
        "efficiency": "Q$_{red}$/Q$_{ox}$",
        "duration": "cycle duration / s",
        "charge": "charge / µC",
        "q_ox": r"$Q_{ox}$",
        "q_red": r"$Q_{red}$",
        "q_red_mean": r"$\langle Q_{red}\rangle$",
        "delta_q": r"$\Delta Q$",
        "leak_fit": r"$\langle i_{par}\rangle t$",
        "k_terms": "k$_{cap}$, k$_{diff}$",
        "k_cap": r"$k_{cap}$ / µF",
        "k_diff": r"$k_{diff}$ / µA (V/s)$^{-1/2}$",
        "i_par": r"$i_{par}$ / µA",
        "current_at": "|i| at {rate} mV/s  /  µA",
        "surface": r"$k_{cap}\cdot\nu$",
        "diffusive": r"$k_{diff}\cdot\sqrt{\nu}$",
        "parasitic": r"$i_{par}$",
        "z_real": "Z′ / Ω",
        "z_imag": "−Z″ / Ω",
        "frequency": "f / Hz",
        "modulus": "|Z| / Ω",
        "modulus_solid": "|Z| / Ω  (solid)",
        "phase": "phase / degrees",
        "phase_dashed": "phase / degrees  (dashed)",
        "c_real": "C′ / µF",
        "c_real_solid": "C′ / µF  (solid)",
        "c_imag_dashed": "C″ / µF  (dashed)",
        "tau": r"$\tau$ / s",
        "gamma": r"$\gamma(\tau)$ / $\gamma_{max}$",
        "gamma_max": "max",
        "residual": "|Z − Z$_{fit}$| / |Z|  /  %",
        "kk_residual": "KK residual / %",
        "measured": "measured",
        "fit": "fit",
        "nyquist": "Nyquist",
        "bode": "Bode",
        "slope45": "45° slope",
        "rs_marker": "▼ R$_s$ from the data",
        "knee_marker": "● knee: phase = −45°",
        "tau0_marker": "■ C″ maximum → τ₀",
        "drt_note": "▲ a pore has no τ of its own",
        "pick_panel": "Choose at least one panel",
        "failed": "{label}\ncould not be drawn",
        "oer": "O$_2$/H$_2$O",
        "her": "H$^+$/H$_2$",
    },
    "es": {
        'time_s': 't / s',
        'gcd_potential': 'E / V vs {reference}',
        'current_density_ma': 'i / mA',
        'current_density_a_g': 'i / A g$^{-1}$',
        'capacitance_uf': 'C / µF',
        'cycle': 'número de ciclo',
        'retention': 'retención de capacitancia / %',
        'coulombic': 'eficiencia culómbica / %',
        'energy_wh': 'E / Wh kg$^{-1}$',
        'power_w': 'P / W kg$^{-1}$',
        'ir_drop_mv': 'caída IR / mV',
        'esr_ohm': 'ESR / $\\Omega$',
        'leakage_ua': 'corriente parásita / µA',
        'capacity_mah': 'Q / mAh g$^{-1}$',
        'overpotential': '$\\eta$ / V',
        'log_j': 'log(|j| / mA cm$^{-2}$)',
        'j_ma_cm2': '|j| / mA cm$^{-2}$',
        'tafel_fit': '{slope} mV dec$^{-1}$',
        'tafel_region': 'región ajustada',
        'asymmetric_note': 'carga y descarga a corrientes distintas',
        'interpolated': 'interpolado de los puntos simétricos',
        'measured_point': 'medido',
        'ceiling': 'techo $\\Delta V^2/4R$',
        'fade_fit': '{model}, R$^2$ = {r2}',
        'benchmark': '{j} mA cm$^{-2}$',
        "potential": "E / V vs {reference}",
        "inverse_c2": "C$^{-2}$ / F$^{-2}$",
        "ms_fit": "{f} Hz: E$_{fb}$ = {efb} V, N = {n} cm$^{-3}$",
        "ms_unused": "fuera de la región lineal",
        "ms_flatband": "E$_{fb}$",
        "potential_plain": "E / V",
        "rhe": "E / V vs RHE",
        "current_mass": "i / A g$^{-1}$",
        "current": "i / A",
        "rate": r"$\nu$ / mV s$^{-1}$",
        "capacitance_mass": "C / F g$^{-1}$",
        "capacitance": "C / F",
        "loop": r"lazo $\div\ 2\nu\Delta V$  (con la fuga)",
        "reduction": r"carga de reducción $\div\ \Delta V$",
        "efficiency": "Q$_{red}$/Q$_{ox}$",
        "duration": "duración del ciclo / s",
        "charge": "carga / µC",
        "q_ox": r"$Q_{ox}$",
        "q_red": r"$Q_{red}$",
        "q_red_mean": r"$\langle Q_{red}\rangle$",
        "delta_q": r"$\Delta Q$",
        "leak_fit": r"$\langle i_{par}\rangle t$",
        "k_terms": "k$_{cap}$, k$_{diff}$",
        "k_cap": r"$k_{cap}$ / µF",
        "k_diff": r"$k_{diff}$ / µA (V/s)$^{-1/2}$",
        "i_par": r"$i_{par}$ / µA",
        "current_at": "|i| a {rate} mV/s  /  µA",
        "surface": r"$k_{cap}\cdot\nu$",
        "diffusive": r"$k_{diff}\cdot\sqrt{\nu}$",
        "parasitic": r"$i_{par}$",
        "z_real": "Z′ / Ω",
        "z_imag": "−Z″ / Ω",
        "frequency": "f / Hz",
        "modulus": "|Z| / Ω",
        "modulus_solid": "|Z| / Ω  (continua)",
        "phase": "fase / grados",
        "phase_dashed": "fase / grados  (discontinua)",
        "c_real": "C′ / µF",
        "c_real_solid": "C′ / µF  (continua)",
        "c_imag_dashed": "C″ / µF  (discontinua)",
        "tau": r"$\tau$ / s",
        "gamma": r"$\gamma(\tau)$ / $\gamma_{máx}$",
        "gamma_max": "máx",
        "residual": "|Z − Z$_{aj}$| / |Z|  /  %",
        "kk_residual": "residuo KK / %",
        "measured": "medido",
        "fit": "ajuste",
        "nyquist": "Nyquist",
        "bode": "Bode",
        "slope45": "pendiente 45°",
        "rs_marker": "▼ R$_s$ de los datos",
        "knee_marker": "● codo: fase = −45°",
        "tau0_marker": "■ máximo de C″ → τ₀",
        "drt_note": "▲ un poro no tiene τ propia",
        "pick_panel": "Elige al menos un panel",
        "failed": "{label}\nno se pudo dibujar",
        "oer": "O$_2$/H$_2$O",
        "her": "H$^+$/H$_2$",
    },
}

#: The four-colour scheme of a published figure: measurement, model,
#: second quantity, and the grey that reference lines are drawn in.
#:
#: Not a categorical palette. These are ROLES — what is measured, what
#: the model says, the quantity on the twin axis — and a figure where
#: "the fitted curve" is the same colour in every panel is a figure that
#: is learned once.
INK = "#1a1a19"
MODEL = "#d4213d"
SECOND = "#6250d6"
GREY = "#8a8a84"

#: Colours for a SERIES: scan rates, samples, temperatures. Okabe–Ito
#: ordered so that consecutive entries stay apart, safe for colour
#: blindness and still separable in greyscale. NOT matplotlib's default
#: cycle.
SERIES_COLOURS = ("#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#d4213d",
                  "#6250d6", "#0072B2", "#CC79A7")

#: Equilibrium potential of the oxygen and hydrogen electrodes on the RHE
#: scale, in volts. They do not move with pH on that scale, which is the
#: whole reason a window is read against RHE.
OER_V_VS_RHE = 1.23
HER_V_VS_RHE = 0.0


def _colour(index: int) -> str:
    return SERIES_COLOURS[index % len(SERIES_COLOURS)]


#: What an axis-scale override can say.
#:
#: ``"auto"`` leaves the panel's own choice alone, which is the default
#: and the right answer most of the time; the other four are applied
#: after the panel has drawn, so they work on every panel without each
#: panel having to know about them.
SCALE_CHOICES: tuple[str, ...] = ("auto", "lineal", "logx", "logy", "loglog")

SCALE_LABELS: tuple[tuple[str, str], ...] = (
    ("auto", "La que pida el panel"),
    ("lineal", "Los dos ejes lineales"),
    ("logx", "Sólo x logarítmico"),
    ("logy", "Sólo y logarítmico"),
    ("loglog", "Los dos ejes logarítmicos"),
)


def apply_scale(axis, choice: str) -> None:
    """Force a panel's axis scales after it has drawn.

    Applied afterwards, and only to the axes that can take it: a
    logarithmic scale on an axis whose data cross zero draws nothing at
    all, and a phase axis running −90 to 0 is exactly that. So each axis
    is checked against its own limits first and left alone when a log
    scale would empty it, which is a refusal rather than a blank panel.
    """
    if choice in ("", "auto"):
        return
    if choice == "lineal":
        axis.set_xscale("linear")
        axis.set_yscale("linear")
        return
    # The test is on the DATA limits and not on the view limits. The
    # view carries matplotlib's autoscale margin, so an axis holding
    # 1 to 100 reports a lower limit of about -4 and a check on that
    # refuses every panel there is. `dataLim` is the bounding box of what
    # was actually plotted.
    box = axis.dataLim
    wanted = (
        (choice in ("logx", "loglog"), axis.set_xscale, box.intervalx),
        (choice in ("logy", "loglog"), axis.set_yscale, box.intervaly),
    )
    for want, setter, interval in wanted:
        if not want:
            continue
        low, high = float(interval[0]), float(interval[1])
        if not (math.isfinite(low) and math.isfinite(high)):
            continue
        # Some positive data is enough: matplotlib masks the
        # non-positive points, which is what a log axis means. All of it
        # non-positive is not — a phase axis running −90 to 0 would come
        # back empty — so that one is refused and left linear.
        if max(low, high) > 0.0:
            setter("log")


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
    panel_height_in: float = 3.0
    dpi: int = 600
    """600 for a raster going into a manuscript; the PDF is resolution
    independent either way."""
    language: str = "en"
    """``"en"`` or ``"es"``. The axis labels only: the window, the report
    and every message stay Spanish."""
    letters: bool = True
    letters_inside: bool = True
    """Bottom-right inside the frame, which is how the figures this
    style came from are drawn. Off puts them above the frame, which never
    collides with anything."""
    legends: bool = True
    grid: bool = False
    reference_axis: bool = True
    """Draw the RHE axis on top of a potential panel, and the O₂ line in
    it, when the electrode declares a pH and a reference. Silently
    skipped when it cannot be computed, because a guessed pH is 59 mV
    per unit of error."""
    font_size: float = 8.5
    title: str = ""
    scales: dict[str, str] = field(default_factory=dict)
    """Axis-scale override per panel key, one of :data:`SCALE_CHOICES`.

    A panel chooses the scale its data want — a frequency axis is
    logarithmic, a potential axis is not — and that is the right default
    and the wrong rule: a Bode plot is read log-log, a Nyquist is
    sometimes drawn log-log to see four decades of a porous electrode at
    once, and a rate series spanning a factor of eight in current reads
    either way depending on what is being argued. Rather than add a
    panel per combination, every panel's scale can be overridden here.
    Missing or ``"auto"`` means whatever the panel asked for."""

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name)
                for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompositeFigure":
        known = set(cls.__dataclass_fields__)
        values = {k: v for k, v in data.items() if k in known}
        if "panels" in values:
            values["panels"] = list(values["panels"])
        if "scales" in values:
            values["scales"] = dict(values["scales"])
        return cls(**values)

    def scale_for(self, key: str) -> str:
        choice = self.scales.get(key, "auto")
        return choice if choice in SCALE_CHOICES else "auto"

    @property
    def rows(self) -> int:
        return max(1, math.ceil(len(self.panels) / max(1, self.columns)))

    def figsize(self) -> tuple[float, float]:
        return (self.width_in, self.panel_height_in * self.rows)

    def text(self, key: str, **kwargs) -> str:
        r"""One label, in the chosen language.

        Substitution is by explicit `replace`, **not** by
        ``str.format``. Almost every label here is LaTeX and LaTeX is
        made of braces: ``$\langle i_{par}\rangle$`` sent through
        ``format`` raises ``KeyError: 'par'``, and the label that did it
        was the one carrying the measured leak current.
        """
        table = LABELS.get(self.language, LABELS["en"])
        value = table.get(key) or LABELS["en"].get(key, key)
        for name, replacement in kwargs.items():
            value = value.replace("{" + name + "}", str(replacement))
        return value


#: Ready-made starting points. The names say what the figure is FOR,
#: because the width and the font size are not independent choices: a
#: figure drawn at 7 inches and shrunk into one column arrives with every
#: label at half size, and that is the commonest way a good figure comes
#: out unreadable.
PRESETS: dict[str, dict[str, Any]] = {
    "articulo_2col": {"width_in": 7.2, "panel_height_in": 3.0, "columns": 2,
                      "font_size": 8.5, "dpi": 600},
    "articulo_1col": {"width_in": 3.4, "panel_height_in": 2.8, "columns": 1,
                      "font_size": 7.5, "dpi": 600},
    "articulo_1col_doble": {"width_in": 3.4, "panel_height_in": 2.4,
                            "columns": 2, "font_size": 6.5, "dpi": 600},
    "presentacion": {"width_in": 10.0, "panel_height_in": 4.4, "columns": 2,
                     "font_size": 12.0, "dpi": 200},
    "presentacion_16_9": {"width_in": 13.3, "panel_height_in": 3.7,
                          "columns": 3, "font_size": 13.0, "dpi": 200},
    "poster": {"width_in": 16.0, "panel_height_in": 6.0, "columns": 2,
               "font_size": 18.0, "dpi": 300},
    "tesis": {"width_in": 6.0, "panel_height_in": 3.2, "columns": 2,
              "font_size": 9.0, "dpi": 400},
    "panel_unico": {"width_in": 3.4, "panel_height_in": 3.0, "columns": 1,
                    "font_size": 8.0, "dpi": 600},
    "cuadricula_2x2": {"width_in": 7.2, "panel_height_in": 3.0, "columns": 2,
                       "font_size": 8.5, "dpi": 600},
    "cuadricula_3x2": {"width_in": 10.5, "panel_height_in": 3.0,
                       "columns": 3, "font_size": 8.5, "dpi": 600},
    "tira_horizontal": {"width_in": 11.0, "panel_height_in": 3.0,
                        "columns": 4, "font_size": 8.0, "dpi": 600},
    "columna_vertical": {"width_in": 3.4, "panel_height_in": 2.4,
                         "columns": 1, "font_size": 7.5, "dpi": 600},
    "borrador": {"width_in": 9.0, "panel_height_in": 3.2, "columns": 2,
                 "font_size": 9.5, "dpi": 110, "grid": True},
}

PRESET_LABELS: tuple[tuple[str, str], ...] = (
    ("articulo_2col", "Artículo, dos columnas (7.2″)"),
    ("articulo_1col", "Artículo, una columna (3.4″)"),
    ("articulo_1col_doble", "Artículo, una columna, dos paneles (3.4″)"),
    ("tesis", "Tesis (6″)"),
    ("presentacion", "Presentación (10″, letra grande)"),
    ("presentacion_16_9", "Presentación 16:9 a tres columnas (13.3″)"),
    ("poster", "Póster (16″, letra 18)"),
    ("panel_unico", "Un solo panel (3.4″)"),
    ("cuadricula_2x2", "Cuadrícula 2×2 (7.2″)"),
    ("cuadricula_3x2", "Cuadrícula 3×2 (10.5″)"),
    ("tira_horizontal", "Tira horizontal de cuatro (11″)"),
    ("columna_vertical", "Columna vertical estrecha (3.4″)"),
    ("borrador", "Borrador para la pantalla (con rejilla)"),
)


def apply_preset(spec: CompositeFigure, name: str) -> bool:
    """Set the layout fields of a preset, keeping the panels chosen.

    A preset says where the figure is GOING and does not touch which
    panels it carries. The fields a preset does not mention are reset to
    the class default rather than left as they were, so that moving off
    the draft preset — the only one that asks for a grid — puts the grid
    back; a preset that leaves a previous one's decisions in place is not
    a preset.
    """
    values = PRESETS.get(name)
    if values is None:
        return False
    blank = CompositeFigure()
    for field_name in ("width_in", "panel_height_in", "columns",
                       "font_size", "dpi", "grid"):
        setattr(spec, field_name,
                values.get(field_name, getattr(blank, field_name)))
    return True


@dataclass(frozen=True)
class PanelSpec:
    """One panel the composite figure can hold."""

    key: str
    label: str
    """What the checkbox says."""
    needs: str
    """What has to be loaded: ``"rates"``, ``"eis"``, ``"cv"``. A panel
    whose data are missing is not offered, rather than offered and then
    drawn empty."""
    draw: Callable[..., None]
    note: str = ""
    """One line on what the panel is for, shown next to the checkbox."""
    technique: str = "cv"
    """Which measurement it belongs to, for the technique filter. With
    every panel of every technique in one list the list is the problem:
    the person who loaded a galvanostatic series should not have to read
    past the voltammetry to find it."""


# -- shared furniture --------------------------------------------------


def _rate_curves(session) -> list:
    return sorted(session.rate_series, key=lambda c: c.scan_rate)


def _eis_list(session) -> list:
    if session.eis_series:
        return list(session.eis_series)
    return [session.eis] if session.eis is not None else []


def _rate_label(curve) -> str:
    """The scan rate as a person would write it.

    Three significant figures, because a rate MEASURED off |dE/dt| comes
    back as 4.99952 mV/s and a legend that says so is reporting the
    clock's precision as if it were a setting.
    """
    return f"{1e3 * curve.scan_rate:.3g} mV/s"


def every_axes(figure) -> list:
    """Every axes a figure holds, **insets included**.

    ``figure.axes`` does not list them: an axes made with
    ``Axes.inset_axes`` is a CHILD of the one it sits in, so it appears
    in that axes' ``child_axes`` and nowhere else. Twins and secondary
    axes do appear, which is why the omission went unnoticed — the grid
    was switched off on five axes out of six and the inset of the
    Nyquist panel kept drawing one.
    """
    out = []
    for axes in figure.axes:
        out.append(axes)
        out.extend(axes.child_axes)
    return out


def _set_grid(axis, spec: CompositeFigure) -> None:
    """Turn the grid on or off, and **never pass line properties with a
    false first argument**.

    ``axis.grid(False, linewidth=0.4)`` does not turn the grid off: it
    turns it ON, with a warning, because matplotlib reads "you gave me
    line properties, so you must want a grid". The suite's matplotlib
    style sets ``axes.grid: True`` for the on-screen figures — right
    there, wrong in a manuscript — so every exported figure came out
    carrying a grid, from code that had asked three times for it to go
    away.
    """
    if spec.grid:
        axis.grid(True, which="major", linewidth=0.4, alpha=0.3)
    else:
        axis.grid(False)


def _tidy(axis, spec: CompositeFigure, log_x: bool = False) -> None:
    """Ticks inwards on all four sides, minor ticks, and the font size.

    Inward ticks on all four sides is the convention of every journal
    this work goes to, and it is applied in one place so a new panel
    cannot forget it.
    """
    from matplotlib.ticker import AutoMinorLocator

    axis.tick_params(which="both", direction="in", top=True, right=True,
                     labelsize=spec.font_size - 0.5)
    if not log_x and axis.get_xscale() == "linear":
        axis.xaxis.set_minor_locator(AutoMinorLocator())
    if axis.get_yscale() == "linear":
        axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.set_xlabel(axis.get_xlabel(), fontsize=spec.font_size)
    axis.set_ylabel(axis.get_ylabel(), fontsize=spec.font_size)
    _set_grid(axis, spec)


def _rhe_shift(session) -> Optional[float]:
    """Volts to add to the measured potential to get RHE, or ``None``.

    ``None`` is a refusal and it travels: without a pH the conversion is
    59 mV per unit of error, so the secondary axis is simply not drawn
    rather than drawn on a guess.
    """
    curve = (session.cv or (session.rate_series[0]
                            if session.rate_series else None)
             or (session.ms_series[0] if session.ms_series else None))
    if curve is None:
        return None
    converted, _ = curve.electrode.to_rhe(np.zeros(1))
    if converted is None:
        return None
    return float(converted[0])


def _reference_axis(axis, session, spec: CompositeFigure) -> None:
    """The RHE axis on top, with the O₂ line where it belongs.

    The second axis is the reason a window means anything: a limit at
    +0.4 V vs Ag/AgCl says nothing by itself, and the same limit at
    +1.455 V vs RHE says it is 225 mV above the oxygen electrode.
    """
    if not spec.reference_axis:
        return
    shift = _rhe_shift(session)
    if shift is None:
        return
    oxygen = OER_V_VS_RHE - shift
    low, high = axis.get_xlim()
    if low <= oxygen <= high:
        axis.axvline(oxygen, color=MODEL, linewidth=0.9,
                     linestyle=(0, (4, 2)), zorder=1)
    hydrogen = HER_V_VS_RHE - shift
    if low <= hydrogen <= high:
        axis.axvline(hydrogen, color=SECOND, linewidth=0.9,
                     linestyle=(0, (4, 2)), zorder=1)
    top = axis.secondary_xaxis(
        "top", functions=(lambda e: e + shift, lambda e: e - shift))
    top.set_xlabel(spec.text("rhe"), fontsize=spec.font_size)
    top.tick_params(which="both", direction="in",
                    labelsize=spec.font_size - 0.5)


def _potential_axis(axis, session, spec: CompositeFigure, curve) -> str:
    """Label the potential axis with the scale the session is drawing.

    ``curve`` is anything carrying an electrode. It used to read
    ``curve.potential`` as well, which made the Mott-Schottky panel —
    whose "curve" is an impedance spectrum, and an ``Impedance`` has no
    potential column — fail after drawing all of its data.
    """
    label = _potential_label(session, spec, curve)
    axis.set_xlabel(label)
    return label


def _potential_label(session, spec: CompositeFigure, curve) -> str:
    """The potential axis's label, without touching any axis.

    Split out because a galvanostatic curve puts the potential on the
    Y axis: the label is the same question and the axis is not.
    """
    electrode = curve.electrode
    _, label, _ = electrode.on_scale(np.zeros(1),
                                     session.potential_scale_choice)
    if session.potential_scale_choice == "medido":
        # The DISPLAY name from the database, not the key: "Ag/AgCl_sat"
        # is how the table is indexed and "Ag/AgCl (KCl sat.)" is what
        # goes on an axis.
        from ..echem.curve import load_echem_database

        key = electrode.reference or ""
        table = load_echem_database().get("reference_electrodes", {})
        reference = table.get(key, {}).get("axis", key)
        label = (spec.text("potential", reference=reference) if reference
                 else spec.text("potential_plain"))
    return label


# -- the voltammetry panels --------------------------------------------


def _panel_cv_series(axis, session, spec, palette) -> None:
    """The voltammogram of the reporting cycle, one per scan rate."""
    curves = _rate_curves(session)
    unit = spec.text("current")
    last = None
    for index, curve in enumerate(curves):
        # The REPORTING cycle's own arrays. Masking the whole file by
        # which potentials appear in the cycle looks equivalent and is
        # not: `isin` matches VALUES, and a voltammogram sweeps the same
        # potentials once per cycle, so every cycle in the file came
        # through and the panel drew the lot.
        cycle, _ = curve.reporting_cycle()
        values, _, _ = cycle.electrode.on_scale(
            cycle.potential, session.potential_scale_choice)
        current = cycle.electrode.specific(cycle.current, "mass")
        if current is None:
            current, unit = cycle.current, spec.text("current")
        else:
            unit = spec.text("current_mass")
        axis.plot(values, current, color=_colour(index), linewidth=1.3,
                  label=_rate_label(curve), zorder=3)
        last = cycle
    axis.set_ylabel(unit)
    axis.axhline(0.0, color=GREY, linewidth=0.6, zorder=1)
    if last is not None:
        _potential_axis(axis, session, spec, last)
    # Headroom so a three-column legend sits above the loops instead of
    # on them.
    low, high = axis.get_ylim()
    axis.set_ylim(low, high + 0.38 * (high - low))
    if spec.legends and curves:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.0,
                    loc="upper left", ncol=3, columnspacing=1.0,
                    handlelength=1.5)
    _tidy(axis, spec)
    _reference_axis(axis, session, spec)


def _panel_cv_capacitance(axis, session, spec, palette) -> None:
    """Both capacitance conventions against ν, with the efficiency on top.

    The efficiency belongs in this panel and not in one of its own: a
    capacitance that falls with ν and an efficiency that rises with it
    are the two faces of the same leak, and separating them invites the
    reader to call the first a transport limitation.
    """
    from matplotlib.ticker import NullFormatter, NullLocator

    from ..echem.cv import (capacitance, coulombic_efficiency,
                            reversible_capacitance)

    curves = _rate_curves(session)
    if not curves:
        return
    per_gram = bool(curves[0].electrode.mass_mg)
    rates, loop, reversible, efficiency = [], [], [], []
    for curve in curves:
        cycle, _ = curve.reporting_cycle()
        # `Capacitance` carries its own specific values, by convention
        # and by basis; it is NOT a bare number.
        value, _ = capacitance(cycle)
        other = reversible_capacitance(cycle)
        rates.append(1e3 * curve.scan_rate)
        loop.append(value.specific_f_per_g if per_gram else value.farads)
        reversible.append(other.specific_f_per_g if per_gram
                          else other.farads)
        efficiency.append(coulombic_efficiency(cycle))
    axis.plot(rates, loop, "o-", color=INK, markersize=5, linewidth=1.3,
              zorder=3, label=spec.text("loop"))
    axis.plot(rates, reversible, "s-", color=SECOND, markersize=5,
              linewidth=1.3, zorder=3, label=spec.text("reduction"))
    twin = axis.twinx()
    twin.plot(rates, efficiency, "^--", color=MODEL, markersize=5,
              linewidth=1.1, zorder=2)
    twin.set_ylabel(spec.text("efficiency"), color=MODEL,
                    fontsize=spec.font_size)
    twin.tick_params(axis="y", colors=MODEL, direction="in",
                     labelsize=spec.font_size - 0.5)
    twin.set_ylim(min(efficiency) - 0.08, 1.04)
    # Logarithmic in ν with the MEASURED rates as the ticks. A rate
    # series is geometric — 5, 10, 20, 50, 100, 200 — so on a linear
    # axis the four slow ones pile up against the origin, which is
    # exactly where the interesting part is.
    axis.set_xscale("log")
    axis.xaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xticks(rates)
    axis.set_xticklabels([f"{rate:.3g}" for rate in rates])
    axis.set_xlabel(spec.text("rate"))
    axis.set_ylabel(spec.text("capacitance_mass") if per_gram
                    else spec.text("capacitance"))
    span = max(loop + reversible) - min(loop + reversible)
    axis.set_ylim(min(loop + reversible) - 0.15 * span,
                  max(loop + reversible) + 0.22 * span)
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.0,
                    loc="upper right", handlelength=1.6)
        axis.annotate(spec.text("efficiency"),
                      xy=(rates[1], efficiency[1]),
                      xycoords=("data", "axes fraction"), xytext=(8, -28),
                      textcoords="offset points",
                      fontsize=spec.font_size - 1.0, color=MODEL)
    _tidy(axis, spec, log_x=True)


def _panel_cv_charge(axis, session, spec, palette) -> None:
    """The charge balance: Q_ox, Q_red and their difference against time.

    ΔQ against the cycle's duration is the measurement of the parasitic
    current — the slope is ⟨i_par⟩ and the intercept has to be zero, so
    the intercept is the check on the model rather than a result. The
    panel shows the two charges as well because that is what makes the
    difference readable: a Q_ox that climbs with the duration while
    Q_red stays flat IS the leak, drawn.
    """
    from ..echem.cv import elapsed_time, integrate_charge

    curves = _rate_curves(session)
    duration, oxidation, reduction, labels = [], [], [], []
    for curve in curves:
        cycle, _ = curve.reporting_cycle()
        time, _ = elapsed_time(cycle)
        anodic, absolute, _ = integrate_charge(cycle)
        duration.append(float(time[-1]))
        oxidation.append(1e6 * anodic)
        reduction.append(1e6 * (absolute - anodic))
        labels.append(_rate_label(curve))
    if not duration:
        return
    duration = np.asarray(duration)
    oxidation = np.asarray(oxidation)
    reduction = np.asarray(reduction)

    study = getattr(session.result, "rates", None)
    leak = getattr(study, "leakage", None) if study else None
    if leak is not None:
        grid = np.linspace(0.0, 1.1 * duration.max(), 50)
        axis.plot(grid, 1e6 * (leak.offset_c + leak.leakage_a * grid),
                  color=MODEL, linewidth=1.3, zorder=2,
                  label=(f"{spec.text('leak_fit')}, "
                         f"{1e6 * leak.leakage_a:.2f} µA, "
                         f"R² = {leak.r_squared:.3f}"))
    axis.axhline(float(reduction.mean()), color=GREY, linewidth=1.0,
                 linestyle=":", zorder=1, label=spec.text("q_red_mean"))
    axis.plot(duration, oxidation, "o", color=INK, markersize=5, zorder=4,
              label=spec.text("q_ox"))
    axis.plot(duration, reduction, "s", color=SECOND, markersize=5,
              zorder=4, label=spec.text("q_red"))
    axis.plot(duration, oxidation - reduction, "^", color=MODEL,
              markersize=5.5, zorder=4, label=spec.text("delta_q"))
    # Only the ends and the middle: one label per point on a series that
    # bunches up at short durations is a smudge, and the axis already
    # says what the duration is.
    for index in {0, len(duration) - 1}:
        x = float(duration[index])
        y = float(oxidation[index] - reduction[index])
        near = x < 0.2 * float(duration.max())
        axis.annotate(labels[index], (x, y), textcoords="offset points",
                      xytext=((7, 11) if near else (-6, 9)),
                      fontsize=spec.font_size - 1.5, color=GREY,
                      ha="left" if near else "right")
    axis.set_xlabel(spec.text("duration"))
    axis.set_ylabel(spec.text("charge"))
    axis.set_xlim(0.0, 1.14 * float(duration.max()))
    axis.set_ylim(0.0, 1.30 * float(oxidation.max()))
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.7,
                    loc="upper left", ncol=2, columnspacing=0.9,
                    handlelength=1.4)
    _tidy(axis, spec)


def _panel_cv_terms(axis, session, spec, palette) -> None:
    """The three coefficients of ``i = k_cap·ν + k_diff·√ν + i_par``.

    With their error bars, because which coefficients the series
    DETERMINES is as much the result as their values: a term whose
    standard error is as large as itself is a term the series cannot
    see.
    """
    study = getattr(session.result, "rates", None)
    models = getattr(study, "full_model", None) if study else None
    if not models:
        return
    potential = np.array([m.potential for m in models])
    axis.errorbar(potential, [1e6 * m.capacitance_f for m in models],
                  yerr=[1e6 * m.errors[0] for m in models],
                  fmt="o-", color=INK, markersize=4.5, linewidth=1.2,
                  capsize=2.5, zorder=3, label=spec.text("k_cap"))
    axis.errorbar(potential, [1e6 * m.diffusive for m in models],
                  yerr=[1e6 * m.errors[1] for m in models],
                  fmt="s-", color=SECOND, markersize=4.5, linewidth=1.2,
                  capsize=2.5, zorder=3, label=spec.text("k_diff"))
    twin = axis.twinx()
    twin.errorbar(potential, [1e6 * m.leakage_a for m in models],
                  yerr=[1e6 * m.errors[2] for m in models],
                  fmt="^--", color=MODEL, markersize=4.5, linewidth=1.1,
                  capsize=2.5, zorder=2)
    # The zero line is the point of the twin axis: i_par changes SIGN
    # across a window that reaches the oxygen and the hydrogen regions,
    # and where it crosses is where the parasitic reaction turns round.
    twin.axhline(0.0, color=MODEL, linewidth=0.6, linestyle=":")
    twin.set_ylabel(spec.text("i_par"), color=MODEL, fontsize=spec.font_size)
    twin.tick_params(axis="y", colors=MODEL, direction="in",
                     labelsize=spec.font_size - 0.5)
    biggest = max(abs(1e6 * m.leakage_a) + abs(1e6 * m.errors[2])
                  for m in models)
    twin.set_ylim(-1.25 * biggest, 1.25 * biggest)
    curve = session.cv or (session.rate_series[0]
                           if session.rate_series else None)
    if curve is not None:
        _potential_axis(axis, session, spec, curve)
    axis.set_ylabel(spec.text("k_terms"))
    tallest = max(1e6 * (abs(m.capacitance_f) + m.errors[0]) for m in models)
    axis.set_ylim(0.0, 1.30 * tallest)
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.0,
                    loc="upper left", handlelength=1.6)
    _tidy(axis, spec)
    _reference_axis(axis, session, spec)


def _panel_cv_stack(axis, session, spec, palette) -> None:
    """The three contributions as a stacked area at the slowest rate.

    The coefficients panel says what the fit measured; this says what it
    MEANS at the rate where the parasitic term matters most, which is
    the slow end.
    """
    study = getattr(session.result, "rates", None)
    models = getattr(study, "full_model", None) if study else None
    if not models or not session.rate_series:
        return
    slowest = min(session.rate_series, key=lambda c: c.scan_rate).scan_rate
    potential = np.array([m.potential for m in models])
    first = np.array([1e6 * abs(m.capacitance_f * slowest) for m in models])
    second = first + np.array([1e6 * abs(m.diffusive * math.sqrt(slowest))
                               for m in models])
    third = second + np.array([1e6 * abs(m.leakage_a) for m in models])
    # Cumulative boundaries drawn as LINES with the bands filled between
    # them, rather than `stackplot`. Visually the same stack; the
    # difference is that a stackplot leaves no Line2D behind, so the
    # panel came out of "save the figure's numbers" with nothing but its
    # header — a figure whose numbers cannot be exported has to be
    # retyped to be used anywhere else.
    for upper, lower, colour, key in (
        (first, np.zeros_like(first), INK, "surface"),
        (second, first, SECOND, "diffusive"),
        (third, second, MODEL, "parasitic"),
    ):
        axis.fill_between(potential, lower, upper, color=colour, alpha=0.75,
                          linewidth=0.0)
        axis.plot(potential, upper, "-", color=colour, linewidth=1.1,
                  label=spec.text(key))
    curve = session.cv or session.rate_series[0]
    _potential_axis(axis, session, spec, curve)
    axis.set_ylabel(spec.text("current_at", rate=f"{1e3 * slowest:.3g}"))
    axis.set_ylim(0.0, 1.30 * float(third.max()))
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.0,
                    loc="upper left", ncol=3, columnspacing=1.0,
                    handlelength=1.4)
    _tidy(axis, spec)
    _reference_axis(axis, session, spec)


def _panel_mott_schottky(axis, session, spec, palette) -> None:
    """``1/C²`` against the DC potential, one line per frequency.

    Every point of every frequency is drawn and the fitted stretch is
    marked, because the linear region is FOUND and a reader has to be
    able to see what was left out. A Mott-Schottky plot shown only over
    the window that was fitted cannot be argued with.
    """
    result = session.mott_schottky()
    if result is None or not result.available:
        return
    for index, fit in enumerate(result.fits):
        colour = _colour(index)
        axis.plot(fit.potentials[~fit.used], fit.inverse_square[~fit.used],
                  "o", color=colour, markersize=3.4, markerfacecolor="none",
                  markeredgewidth=0.9, alpha=0.5, zorder=2)
        axis.plot(fit.potentials[fit.used], fit.inverse_square[fit.used],
                  "o", color=colour, markersize=4.2, zorder=3)
        grid = np.linspace(min(fit.window[0], fit.flat_band_v),
                           fit.window[1], 32)
        axis.plot(grid, fit.slope * grid + fit.intercept, "-", color=colour,
                  linewidth=1.2, zorder=2,
                  label=spec.text("ms_fit",
                                  f=f"{fit.frequency_hz:.4g}",
                                  efb=f"{fit.flat_band_v:+.3f}",
                                  n=f"{fit.carrier_density_cm3:.2g}"))
        axis.plot([fit.flat_band_v], [0.0], "v", color=colour, markersize=6,
                  zorder=4)
    axis.axhline(0.0, color=GREY, linewidth=0.7, zorder=1)
    curve = session.ms_series[0] if session.ms_series else None
    if curve is not None:
        _potential_axis(axis, session, spec, curve)
    axis.set_ylabel(spec.text("inverse_c2"))
    axis.set_ylim(0.0, None)
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.5,
                    loc="upper left", handlelength=1.4)
    axis.text(0.03, 0.05, f"▼ {spec.text('ms_flatband')}",
              transform=axis.transAxes, ha="left",
              fontsize=spec.font_size - 1.5, color=GREY)
    _tidy(axis, spec)
    _reference_axis(axis, session, spec)


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


def _nyquist_limit(spectra) -> float:
    """Where to cut the high-frequency corner of a Nyquist plot.

    Six times the largest series resistance. Bounded by the RESISTANCES
    and not by the data: one decade of frequency more puts the capacitive
    tail on the axis and flattens the corner, which is the part the
    figure is for.
    """
    return 6.0 * max(float(s.z.real.min()) for s in spectra)


def _panel_nyquist(axis, session, spec, palette) -> None:
    """Nyquist at EQUAL aspect, with the high-frequency corner inset.

    The aspect is not cosmetic and it is the reason this panel is drawn
    this way rather than with free limits: a Nyquist plot is read by its
    shape, so a box whose axes run 0–1400 against 0–8000 draws the 45°
    region of a porous electrode almost vertical, and anyone reading a
    depression angle or a transport regime off it is reading the axes.
    The inset is how the field shows the corner without giving up the
    aspect: it is the commonest supercapacitor figure there is.
    """
    spectra = _eis_list(session)
    analyses = session.eis_analyses()
    if not spectra:
        return
    high = max(float(max(s.z.real.max(), (-s.z.imag).max()))
               for s in spectra) * 1.05
    for index, (spectrum, analysis) in enumerate(zip(spectra, analyses)):
        axis.plot(spectrum.z.real, -spectrum.z.imag, "o",
                  color=_colour(index), markersize=3.2,
                  markerfacecolor="none", markeredgewidth=0.9, zorder=3,
                  label=spectrum.name)
        fit = analysis.fit if analysis else None
        if fit is not None and fit.fitted is not None:
            axis.plot(fit.fitted.real, -fit.fitted.imag, "-",
                      color=_colour(index), linewidth=1.2, zorder=2)
    axis.set_xlim(0.0, high)
    axis.set_ylim(0.0, high)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(spec.text("z_real"))
    axis.set_ylabel(spec.text("z_imag"))
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.0,
                    loc="upper left", handlelength=1.2, borderpad=0.2)
    _tidy(axis, spec)

    limit = _nyquist_limit(spectra)
    if limit <= 0.0 or limit >= high:
        return
    inset = axis.inset_axes((0.54, 0.17, 0.42, 0.42))
    for index, (spectrum, analysis) in enumerate(zip(spectra, analyses)):
        keep = -spectrum.z.imag <= limit
        inset.plot(spectrum.z.real[keep], -spectrum.z.imag[keep], "o",
                   color=_colour(index), markersize=2.4,
                   markerfacecolor="none", markeredgewidth=0.7, zorder=3)
        fit = analysis.fit if analysis else None
        if fit is not None and fit.fitted is not None:
            inset.plot(fit.fitted.real[keep], -fit.fitted.imag[keep], "-",
                       color=_colour(index), linewidth=1.0, zorder=2)
        if analysis is not None and analysis.series_resistance is not None:
            inset.plot([analysis.series_resistance], [0.0], "v",
                       color=_colour(index), markersize=4, zorder=4)
    inset.plot([0.0, limit], [0.0, limit], linestyle=(0, (3, 2)),
               color=GREY, linewidth=0.8, zorder=1)
    inset.set_xlim(0.0, limit)
    inset.set_ylim(0.0, limit)
    inset.set_aspect("equal", adjustable="box")
    inset.tick_params(which="both", direction="in", top=True, right=True,
                      labelsize=spec.font_size - 2.5)


def _panel_nyquist_corner(axis, session, spec, palette) -> None:
    """The high-frequency corner on its own, linear and square."""
    spectra = _eis_list(session)
    analyses = session.eis_analyses()
    if not spectra:
        return
    limit = _nyquist_limit(spectra)
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
    axis.plot([0.0, limit], [0.0, limit], linestyle=(0, (4, 2)), color=GREY,
              linewidth=0.9, zorder=1)
    axis.text(0.80 * limit, 0.86 * limit, spec.text("slope45"), color=GREY,
              fontsize=spec.font_size - 1.5, rotation=45, ha="center",
              va="center")
    axis.set_xlim(0.0, limit)
    axis.set_ylim(0.0, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(spec.text("z_real"))
    axis.set_ylabel(spec.text("z_imag"))
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.5,
                    loc="upper left", handlelength=1.2, borderpad=0.2)
    axis.text(0.03, 0.05, spec.text("rs_marker"), transform=axis.transAxes,
              ha="left", fontsize=spec.font_size - 1.5, color=GREY)
    _tidy(axis, spec)


def _panel_nyquist_bode(axis, session, spec, palette) -> None:
    """Nyquist and Bode phase in ONE box: frequency on top, phase right.

    The compact presentation several groups use, offered because people
    use it — and with the caveat that goes with it. The two curves share
    a frame and nothing else: the Nyquist runs against Z′ on the bottom
    axis and the phase against f on the top one, so a point of one has
    no counterpart in the other and the eye will try to find one anyway.
    The Nyquist here also cannot be at equal aspect, since the box is
    shared, so the 45° region is drawn at whatever angle the limits give.
    :func:`_panel_nyquist` keeps the aspect and insets the corner, which
    is what this one trades away.
    """
    spectra = _eis_list(session)
    analyses = session.eis_analyses()
    if not spectra:
        return
    # Three axes, not two, and that is the catch of this format.
    # `axis.twiny()` keeps the LEFT y, which here is −Z″ in ohms, so the
    # phase (−90 to 0) drew as a flat line on the floor of the Nyquist.
    # The phase needs its own y AND its own x: a twinx for the y, then a
    # twiny OF THAT for the x.
    phase = axis.twinx()
    top = phase.twiny()
    top.set_xscale("log")
    # High frequency on the LEFT, which is how a Bode plot is read and
    # how the sample figures this follows are drawn.
    top.set_xlim(max(float(s.frequency.max()) for s in spectra),
                 min(float(s.frequency.min()) for s in spectra))
    for index, (spectrum, analysis) in enumerate(zip(spectra, analyses)):
        colour = _colour(index)
        name = spectrum.name if len(spectra) > 1 else ""
        axis.plot(spectrum.z.real, -spectrum.z.imag, "s-", color=colour,
                  markersize=3.0, linewidth=0.9, zorder=3,
                  label=f"{spec.text('nyquist')}{f' — {name}' if name else ''}")
        top.plot(spectrum.frequency, spectrum.phase_deg, "o", color=colour,
                 markersize=3.4, markerfacecolor=colour, alpha=0.75,
                 zorder=2,
                 label=f"{spec.text('bode')}{f' — {name}' if name else ''}")
    axis.set_xlabel(spec.text("z_real"))
    axis.set_ylabel(spec.text("z_imag"))
    top.set_xlabel(spec.text("frequency"), fontsize=spec.font_size)
    top.tick_params(which="both", direction="in",
                    labelsize=spec.font_size - 0.5)
    phase.set_ylabel(spec.text("phase"), fontsize=spec.font_size)
    # Phase increasing DOWNWARDS, so −90° is at the top: that is how the
    # reference figures for this format are drawn, and it puts the
    # capacitive end of the sweep next to the capacitive end of the
    # Nyquist instead of mirrored against it.
    phase.set_ylim(5.0, -95.0)
    phase.tick_params(which="both", direction="in",
                      labelsize=spec.font_size - 0.5)
    top.set_ylim(*phase.get_ylim())
    axis.set_xlim(0.0, None)
    axis.set_ylim(0.0, None)
    if spec.legends:
        handles, labels = axis.get_legend_handles_labels()
        extra = top.get_legend_handles_labels()
        # Anchored just above the floor so it clears the panel letter,
        # which sits in that corner.
        axis.legend(handles + extra[0], labels + extra[1], frameon=False,
                    fontsize=spec.font_size - 1.5, loc="lower right",
                    bbox_to_anchor=(1.0, 0.10), handlelength=1.4)
    axis.tick_params(which="both", direction="in",
                     labelsize=spec.font_size - 0.5)
    axis.set_xlabel(axis.get_xlabel(), fontsize=spec.font_size)
    axis.set_ylabel(axis.get_ylabel(), fontsize=spec.font_size)


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
    phase.axhline(KNEE_PHASE_DEG, color=GREY, linewidth=0.7,
                  linestyle=(0, (1, 2)), zorder=1)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(spec.text("frequency"))
    axis.set_ylabel(spec.text("modulus_solid"))
    phase.set_ylabel(spec.text("phase_dashed"), fontsize=spec.font_size)
    phase.set_ylim(-95.0, 5.0)
    phase.tick_params(which="both", direction="in",
                      labelsize=spec.font_size - 0.5)
    phase.text(0.97, 0.30, spec.text("knee_marker"),
               transform=phase.transAxes, ha="right", va="bottom",
               fontsize=spec.font_size - 1.5, color=GREY)
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.0,
                    loc="lower left", handlelength=1.2)
    _tidy(axis, spec, log_x=True)


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
    axis.set_xlabel(spec.text("frequency"))
    axis.set_ylabel(spec.text("c_real_solid"))
    second.set_ylabel(spec.text("c_imag_dashed"), fontsize=spec.font_size)
    second.tick_params(which="both", direction="in",
                       labelsize=spec.font_size - 0.5)
    second.text(0.97, 0.62, spec.text("tau0_marker"),
                transform=second.transAxes, ha="right",
                fontsize=spec.font_size - 1.5, color=GREY, va="top")
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.0,
                    handlelength=1.2)
    _tidy(axis, spec, log_x=True)


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
                  label=f"{spectrum.name}  "
                        f"({spec.text('gamma_max')} {peak:.0f} Ω)")
        for tau in result.peaks_s:
            axis.plot([tau], [0.0], "^", color=_colour(index), markersize=5,
                      clip_on=False, zorder=5)
    axis.set_xscale("log")
    axis.set_ylim(0.0, 1.18)
    axis.set_xlabel(spec.text("tau"))
    axis.set_ylabel(spec.text("gamma"))
    axis.text(0.03, 0.46, spec.text("drt_note"), transform=axis.transAxes,
              ha="left", va="top", fontsize=spec.font_size - 1.5, color=GREY)
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.5,
                    loc="upper left", handlelength=1.2)
    _tidy(axis, spec, log_x=True)


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
    axis.axhline(0.0, color=GREY, linewidth=0.6, zorder=1)
    axis.set_xscale("log")
    axis.set_xlabel(spec.text("frequency"))
    axis.set_ylabel(spec.text("residual"))
    low, high = axis.get_ylim()
    axis.set_ylim(low, high + 0.35 * (high - low))
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.5,
                    loc="upper right", handlelength=1.3)
    _tidy(axis, spec, log_x=True)


def _panel_kk(axis, session, spec, palette) -> None:
    for index, (spectrum, analysis) in enumerate(
            zip(_eis_list(session), session.eis_analyses())):
        kk = analysis.kk if analysis else None
        if kk is None:
            continue
        axis.semilogx(spectrum.frequency, 100.0 * kk.residual_real, "o-",
                      color=_colour(index), markersize=2.5, linewidth=0.7,
                      zorder=3,
                      label=f"{spectrum.name} (Z′, "
                            f"{'ok' if kk.passes else '✗'})")
        axis.semilogx(spectrum.frequency, 100.0 * kk.residual_imag,
                      linestyle=(0, (3, 2)), marker="s", markersize=2.5,
                      linewidth=0.7, color=_colour(index), zorder=2)
    axis.axhline(0.0, color=GREY, linewidth=0.6, zorder=1)
    axis.set_xlabel(spec.text("frequency"))
    axis.set_ylabel(spec.text("kk_residual"))
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.5,
                    loc="upper right", handlelength=1.3)
    _tidy(axis, spec, log_x=True)


# -- galvanostatic -----------------------------------------------------


def _gcd_pairs(session):
    """``(curve, analysis)`` for every charge-discharge curve loaded."""
    return [(curve, analysis) for curve, analysis
            in zip(session.gcd_curves(), session.gcd_analyses())
            if analysis is not None]


def _panel_gcd_series(axis, session, spec, palette) -> None:
    """The charge-discharge curves themselves, one per current.

    Each on its OWN time origin rather than on the file's clock: the
    five curves of a rate study were measured one after another over
    sixteen hours, so on a shared clock they appear side by side instead
    of on top of one another and the comparison the panel exists for is
    not drawn at all.
    """
    from ..echem.gcd import half_cycles

    for index, (curve, analysis) in enumerate(_gcd_pairs(session)):
        halves = half_cycles(curve)
        if not halves:
            continue
        # The reported cycle, which is the last pair by default: the
        # first cycles of a GCPL are formation, not performance.
        wanted = analysis.discharges[-1].index if analysis.discharges else -1
        pair = [h for h in halves if h.index in (wanted - 1, wanted)]
        if not pair:
            pair = halves[-2:]
        start = float(pair[0].time[0])
        time = np.concatenate([h.time for h in pair]) - start
        potential = np.concatenate([h.potential for h in pair])
        axis.plot(time, potential, "-", color=_colour(index), linewidth=1.3,
                  zorder=3, label=session.gcd_label(curve)
                  + (" *" if analysis.asymmetric else ""))
    axis.set_xlabel(spec.text("time_s"))
    pairs = _gcd_pairs(session)
    if pairs:
        # The potential is on the Y axis here, so the label and not the
        # axis helper. And no RHE axis: that one draws on top, which is
        # the time axis in this panel.
        axis.set_ylabel(_potential_label(session, spec, pairs[0][0]))
    if any(a.asymmetric for _, a in pairs):
        # Bottom LEFT: the panel letter sits bottom right by default and
        # the two were drawing on top of each other.
        axis.text(0.03, 0.97, "* " + spec.text("asymmetric_note"),
                  transform=axis.transAxes, ha="left", va="top",
                  fontsize=spec.font_size - 2.0, color=GREY)
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.0,
                    loc="upper right", handlelength=1.4)
    _tidy(axis, spec)


def _panel_gcd_rate(axis, session, spec, palette) -> None:
    """Capacitance against discharge current: the rate capability.

    The asymmetric points are drawn as open symbols and left out of the
    line, because a line through them says they belong on the same axis
    and they do not: a pair with two currents has no single current
    density. What the symmetric points interpolate to at the same
    current is drawn as a cross, so the two can be compared.
    """
    pairs = _gcd_pairs(session)
    clean_x, clean_y, broken_x, broken_y = [], [], [], []
    per_gram = bool(session.electrode.mass_mg)
    for curve, analysis in pairs:
        branch = analysis.discharges[-1] if analysis.discharges else None
        if branch is None or branch.capacitance_f is None:
            continue
        from .echem_state import _discharge_current
        current = 1e6 * _discharge_current(curve)
        value = (analysis.specific_f_per_g if per_gram
                 else 1e6 * branch.capacitance_f)
        if value is None:
            continue
        if analysis.asymmetric:
            broken_x.append(current)
            broken_y.append(value)
        else:
            clean_x.append(current)
            clean_y.append(value)
    order = np.argsort(clean_x) if clean_x else []
    if len(clean_x):
        axis.plot(np.asarray(clean_x)[order], np.asarray(clean_y)[order],
                  "o-", color=INK, markersize=5, linewidth=1.3, zorder=3,
                  label=spec.text("measured_point"))
    if broken_x:
        axis.plot(broken_x, broken_y, "o", markerfacecolor="none",
                  markeredgecolor=INK, markersize=6, linewidth=0, zorder=4,
                  label=spec.text("asymmetric_note"))
        for label, milliamps, measured, estimate in session.gcd_interpolated():
            scale = 1.0 if per_gram else 1.0
            axis.plot([1e3 * milliamps], [estimate * scale], "x",
                      color=MODEL, markersize=7, markeredgewidth=1.4,
                      zorder=5, label=spec.text("interpolated"))
    axis.set_xscale("log")
    axis.set_xlabel("i / µA")
    axis.set_ylabel(spec.text("capacitance_mass") if per_gram
                    else spec.text("capacitance_uf"))
    if spec.legends and (broken_x or clean_x):
        handles, labels = axis.get_legend_handles_labels()
        seen: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            seen.setdefault(label, handle)
        axis.legend(seen.values(), seen.keys(), frameon=False,
                    fontsize=spec.font_size - 1.5, loc="upper right",
                    handlelength=1.4)
    _tidy(axis, spec, log_x=True)


def _panel_gcd_cycling(axis, session, spec, palette) -> None:
    """Capacitance retention cycle by cycle, one curve per current.

    This is the panel the section could not draw at all: a GCPL keeps
    its cycles inside one file and the analysis gave one number per
    file, so a thousand cycles of a measurement arrived as a point.
    """
    for index, (curve, cycling) in enumerate(
            zip(session.gcd_curves(), session.gcd_cyclings())):
        if cycling is None or not cycling.capacitance_f:
            continue
        values = np.asarray(cycling.capacitance_f, dtype=float)
        if values[0] <= 0:
            continue
        axis.plot(cycling.cycles, 100.0 * values / values[0], "-",
                  color=_colour(index), linewidth=1.2, zorder=3,
                  label=session.gcd_label(curve))
    axis.axhline(100.0, color=GREY, linewidth=0.7, linestyle=(0, (1, 2)),
                 zorder=1)
    axis.set_xlabel(spec.text("cycle"))
    axis.set_ylabel(spec.text("retention"))
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.0,
                    loc="lower left", handlelength=1.4)
    _tidy(axis, spec)


def _panel_gcd_efficiency(axis, session, spec, palette) -> None:
    """Coulombic efficiency cycle by cycle.

    Above 100 % is charge coming out that did not go in, and over a
    thousand cycles it is not a single cycle's noise. The panel is drawn
    with the 100 % line because the eye has to see which side of it the
    curve sits on.
    """
    for index, (curve, cycling) in enumerate(
            zip(session.gcd_curves(), session.gcd_cyclings())):
        if cycling is None or not cycling.efficiency:
            continue
        axis.plot(cycling.cycles[:len(cycling.efficiency)],
                  100.0 * np.asarray(cycling.efficiency, dtype=float), "-",
                  color=_colour(index), linewidth=1.2, zorder=3,
                  label=session.gcd_label(curve))
    axis.axhline(100.0, color=GREY, linewidth=0.8, zorder=1)
    axis.set_xlabel(spec.text("cycle"))
    axis.set_ylabel(spec.text("coulombic"))
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.0,
                    loc="upper right", handlelength=1.4)
    _tidy(axis, spec)


def _panel_gcd_leakage(axis, session, spec, palette) -> None:
    """The parasitic current against the applied one.

    The point of the panel is that it should be FLAT. A coulombic excess
    cannot be compared between currents — the same leak is a large
    percentage at a small current and a small one at a large current —
    and dividing it by the cycle time turns it into a current that can.
    On the user's five files it comes out at 0.19 to 0.25 µA across a
    factor of eight in applied current, the asymmetric one included,
    and a flat line is that statement drawn.
    """
    from .echem_state import _discharge_current

    currents, leaks, broken_x, broken_y = [], [], [], []
    for curve, analysis in _gcd_pairs(session):
        if analysis.leakage_a is None:
            continue
        current = 1e6 * _discharge_current(curve)
        value = 1e6 * analysis.leakage_a
        if analysis.asymmetric:
            broken_x.append(current)
            broken_y.append(value)
        else:
            currents.append(current)
            leaks.append(value)
    if currents:
        order = np.argsort(currents)
        axis.plot(np.asarray(currents)[order], np.asarray(leaks)[order],
                  "o-", color=INK, markersize=5, linewidth=1.3, zorder=3,
                  label=spec.text("measured_point"))
        mean = float(np.mean(leaks))
        axis.axhline(mean, color=MODEL, linewidth=0.9,
                     linestyle=(0, (4, 2)), zorder=2,
                     label=f"{mean:.3g} µA")
    if broken_x:
        axis.plot(broken_x, broken_y, "o", markerfacecolor="none",
                  markeredgecolor=INK, markersize=6, linewidth=0, zorder=4,
                  label=spec.text("asymmetric_note"))
    axis.set_xscale("log")
    axis.set_xlabel("i / µA")
    axis.set_ylabel(spec.text("leakage_ua"))
    if spec.legends and (currents or broken_x):
        axis.legend(frameon=False, fontsize=spec.font_size - 1.5,
                    loc="upper left", handlelength=1.6)
    _tidy(axis, spec, log_x=True)


def _panel_gcd_resistance(axis, session, spec, palette) -> None:
    """The IR jump and the resistance it implies, against current.

    Two scales on purpose: the jump grows with the current and the
    resistance should not, so a flat ESR next to a rising jump is the
    check that the jump really is ohmic. The denominator is the CHANGE
    in current, which is the sum of the two magnitudes and equals 2|I|
    only on a symmetric cycle — getting that wrong doubled the
    resistance of the user's asymmetric file.
    """
    from .echem_state import _discharge_current

    currents, drops, resistances, marks = [], [], [], []
    for curve, analysis in _gcd_pairs(session):
        branch = analysis.discharges[-1] if analysis.discharges else None
        if branch is None or not branch.ir_drop_v:
            continue
        currents.append(1e6 * _discharge_current(curve))
        drops.append(1e3 * branch.ir_drop_v)
        resistances.append(analysis.resistance_ohm or np.nan)
        marks.append(analysis.asymmetric)
    if not currents:
        return
    order = np.argsort(currents)
    current_array = np.asarray(currents)[order]
    axis.plot(current_array, np.asarray(drops)[order], "o-", color=INK,
              markersize=5, linewidth=1.3, zorder=3)
    axis.set_xscale("log")
    axis.set_xlabel("i / µA")
    axis.set_ylabel(spec.text("ir_drop_mv"), color=INK)
    twin = axis.twinx()
    twin.plot(current_array, np.asarray(resistances)[order], "s--",
              color=MODEL, markersize=5, linewidth=1.1, zorder=2)
    twin.set_ylabel(spec.text("esr_ohm"), color=MODEL,
                    fontsize=spec.font_size)
    twin.tick_params(axis="y", colors=MODEL, direction="in",
                     labelsize=spec.font_size - 0.5)
    finite = np.asarray(resistances, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        twin.set_ylim(0.0, 1.6 * float(finite.max()))
    _tidy(axis, spec, log_x=True)


def _panel_ragone(axis, session, spec, palette) -> None:
    """Energy against power, log-log, with the matched-load ceiling.

    Both per KILOGRAM of active material of a SINGLE electrode, which is
    the basis that has to be stated: a packaged cell carries two
    electrodes, electrolyte, separator and collectors, and its energy
    per total mass is three to five times lower. The ceiling is
    ``ΔV²/4R`` and no measured discharge reaches it, which is why it is
    a line and not a point.
    """
    points = session.ragone_points()
    if points:
        energies = [e for e, _, _ in points]
        powers = [p for _, p, _ in points]
        order = np.argsort(powers)
        axis.plot(np.asarray(powers)[order], np.asarray(energies)[order],
                  "o-", color=INK, markersize=6, linewidth=1.3, zorder=3)
        for energy, power, label in points:
            axis.annotate(label, xy=(power, energy), xytext=(4, 5),
                          textcoords="offset points",
                          fontsize=spec.font_size - 2.0, color=GREY)
    ceilings = [a.max_power_w_per_kg for _, a in _gcd_pairs(session)
                if a.max_power_w_per_kg]
    if ceilings and points:
        ceiling = float(np.mean(ceilings))
        reach = max(p for _, p, _ in points)
        if ceiling <= 100.0 * reach:
            axis.axvline(ceiling, color=MODEL, linewidth=0.9,
                         linestyle=(0, (4, 2)), zorder=2,
                         label=spec.text("ceiling"))
            if spec.legends:
                axis.legend(frameon=False, fontsize=spec.font_size - 1.5,
                            loc="lower left", handlelength=1.6)
        else:
            # More than two decades past the fastest measured discharge.
            # Drawn, it would squeeze every measured point into the left
            # edge to show a line nothing reaches, so the number goes in
            # as text and the axis stays the data's.
            axis.text(0.03, 0.06,
                      f"{spec.text('ceiling')}: {ceiling:.3g} W kg$^{{-1}}$",
                      transform=axis.transAxes, ha="left", va="bottom",
                      fontsize=spec.font_size - 1.5, color=MODEL)
    axis.set_xscale("log")
    axis.set_yscale("log")
    # Room for the point labels: on a log axis the automatic margin is
    # in display units and the annotations are not, so the first and
    # last points sit on the frame with their labels outside it.
    if points:
        powers = [p for _, p, _ in points]
        energies = [e for e, _, _ in points]
        axis.set_xlim(min(powers) / 2.2, max(powers) * 2.2)
        axis.set_ylim(min(energies) / 1.35, max(energies) * 1.35)
    axis.set_xlabel(spec.text("power_w"))
    axis.set_ylabel(spec.text("energy_wh"))
    _tidy(axis, spec, log_x=True)


# -- catalysis ---------------------------------------------------------


def _catalysis(session):
    result = session.result
    return getattr(result, "catalysis", None) if result is not None else None


def _panel_her_polarisation(axis, session, spec, palette) -> None:
    """The polarisation curve itself, η against |j|.

    Drawn as well as the Tafel plot and not instead of it: a Tafel plot
    is a logarithm and hides the SHAPE, and the shape is where a
    mass-transport limit or a bubble problem shows up. The benchmark
    current densities are marked where the curve reaches them, and the
    ones it does not reach are simply not marked.
    """
    catalysis = _catalysis(session)
    if catalysis is None or catalysis.curve_j is None:
        return
    axis.plot(np.abs(catalysis.curve_j), catalysis.curve_eta, "-",
              color=INK, linewidth=1.4, zorder=3,
              label=spec.text("her") if session.reaction == "HER"
              else spec.text("oer"))
    for density, overpotential in (catalysis.overpotentials or {}).items():
        if overpotential is None:
            continue
        axis.plot([density], [overpotential], "o", color=MODEL,
                  markersize=5, zorder=4)
        axis.annotate(spec.text("benchmark", j=f"{density:g}"),
                      xy=(density, overpotential), xytext=(5, -9),
                      textcoords="offset points",
                      fontsize=spec.font_size - 2.0, color=MODEL)
    axis.set_xlabel(spec.text("j_ma_cm2"))
    axis.set_ylabel(spec.text("overpotential"))
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.0,
                    loc="lower right", handlelength=1.4)
    _tidy(axis, spec)


def _panel_her_tafel(axis, session, spec, palette) -> None:
    """η against log|j|, with the region the slope was FITTED over.

    The fitted region is drawn because a Tafel slope quoted over less
    than a decade is not a Tafel slope, and the only way a reader can
    judge that is to see how much of the curve the line covers.
    """
    catalysis = _catalysis(session)
    if catalysis is None or catalysis.curve_j is None:
        return
    density = np.abs(np.asarray(catalysis.curve_j, dtype=float))
    overpotential = np.asarray(catalysis.curve_eta, dtype=float)
    keep = density > 0
    if not np.any(keep):
        return
    logj = np.log10(density[keep])
    eta = overpotential[keep]
    axis.plot(logj, eta, "-", color=INK, linewidth=1.3, zorder=3,
              label=spec.text("measured"))
    tafel = catalysis.tafel
    if tafel is not None and tafel.valid:
        low, high = tafel.range_mv
        inside = (eta >= min(low, high) * 1e-3) & (eta <= max(low, high) * 1e-3)
        if np.any(inside):
            axis.plot(logj[inside], eta[inside], "-", color=MODEL,
                      linewidth=2.6, alpha=0.45, zorder=2,
                      label=spec.text("tafel_region"))
        axis.text(0.04, 0.92,
                  spec.text("tafel_fit",
                            slope=f"{tafel.slope_mv_per_decade:.0f}"),
                  transform=axis.transAxes, ha="left", va="top",
                  fontsize=spec.font_size - 0.5, color=MODEL)
    axis.set_xlabel(spec.text("log_j"))
    axis.set_ylabel(spec.text("overpotential"))
    if spec.legends:
        axis.legend(frameon=False, fontsize=spec.font_size - 1.5,
                    loc="lower right", handlelength=1.6)
    _tidy(axis, spec)


#: Every panel the composite figure can hold, in the order they are
#: offered. The order is the order a reader goes through them: the
#: measurement first, then what was derived from it, then what checks it.
ECHEM_PANELS: tuple[PanelSpec, ...] = (
    PanelSpec("cv_serie", "CV: la serie de velocidades", "rates",
              _panel_cv_series,
              "los voltamperogramas del ciclo elegido, con el eje RHE y la "
              "línea del O₂ arriba",
              technique="cv"),
    PanelSpec("cv_capacitancia", "CV: capacitancia y eficiencia contra ν",
              "rates", _panel_cv_capacitance,
              "los dos convenios de capacitancia en ν logarítmica, con "
              "Q_red/Q_ox encima: las dos caras de la misma fuga",
              technique="cv"),
    PanelSpec("cv_carga", "CV: el balance de carga", "rates",
              _panel_cv_charge,
              "Q_ox, Q_red y ΔQ contra la duración; la pendiente es la "
              "corriente parásita y la ordenada tiene que ser cero",
              technique="cv"),
    PanelSpec("cv_terminos", "CV: k_cap, k_diff e i_par con sus errores",
              "rates", _panel_cv_terms,
              "los tres coeficientes potencial a potencial; cuáles están "
              "DETERMINADOS es tanto resultado como sus valores",
              technique="cv"),
    PanelSpec("cv_pila", "CV: las tres contribuciones apiladas", "rates",
              _panel_cv_stack,
              "qué fracción de la corriente se lleva cada término a la "
              "velocidad más lenta",
              technique="cv"),
    PanelSpec("cv_dunn", "CV: la separación de Dunn", "cv", _panel_dunn,
              "el ciclo cerrado con la parte superficial sembrada dentro",
              technique="cv"),
    PanelSpec("mott_schottky", "Mott-Schottky: 1/C² contra E", "ms",
              _panel_mott_schottky,
              "una recta por frecuencia, con la región lineal marcada y "
              "el E_fb de cada una en el eje",
              technique="ms"),
    PanelSpec("eis_nyquist", "EIS: Nyquist cuadrado con recuadro", "eis",
              _panel_nyquist,
              "a escala uno a uno, con el rincón de alta frecuencia en un "
              "recuadro. La figura más usada del campo",
              technique="eis"),
    PanelSpec("eis_rincon", "EIS: sólo el rincón de alta frecuencia", "eis",
              _panel_nyquist_corner,
              "el mismo recuadro en un panel entero, con la R_s medida",
              technique="eis"),
    PanelSpec("eis_nyquist_bode", "EIS: Nyquist y Bode en la misma caja",
              "eis", _panel_nyquist_bode,
              "el formato compacto: frecuencia arriba y fase a la derecha. "
              "Ojo, el Nyquist ahí NO puede ir a escala uno a uno",
              technique="eis"),
    PanelSpec("eis_bode", "EIS: Bode con el codo marcado", "eis",
              _panel_bode,
              "|Z| y la fase, con el cruce de −45° de cada espectro",
              technique="eis"),
    PanelSpec("eis_capacitancia", "EIS: C′(f) y C″(f)", "eis",
              _panel_complex_capacitance,
              "la capacitancia sacada de Z sin ajustar nada, y τ₀ en el "
              "máximo de C″",
              technique="eis"),
    PanelSpec("eis_drt", "EIS: la DRT normalizada", "eis",
              _panel_drt_series,
              "γ(τ) de cada espectro sobre su propio máximo, sólo en el "
              "intervalo medido",
              technique="eis"),
    PanelSpec("eis_residuo", "EIS: el residuo del ajuste", "eis",
              _panel_eis_residual,
              "dónde falla el circuito, frecuencia a frecuencia",
              technique="eis"),
    PanelSpec("eis_kk", "EIS: el residuo de Kramers-Kronig", "eis",
              _panel_kk,
              "se lee la FORMA y no el tamaño: rachas de un signo son "
              "deriva de la celda",
              technique="eis"),
    PanelSpec("gcd_series", "GCD: las curvas de carga-descarga", "gcd",
              _panel_gcd_series,
              "una por corriente, cada una en su propio origen de "
              "tiempo; las asimétricas van marcadas",
              technique="gcd"),
    PanelSpec("gcd_velocidad", "GCD: capacitancia contra corriente", "gcd",
              _panel_gcd_rate,
              "la capacidad de velocidad, con los puntos asimétricos en "
              "símbolo hueco y fuera de la línea",
              technique="gcd"),
    PanelSpec("gcd_ciclado", "GCD: retención ciclo a ciclo", "gcd",
              _panel_gcd_cycling,
              "mil ciclos por archivo, que es lo que la sección no podía "
              "dibujar en absoluto",
              technique="gcd"),
    PanelSpec("gcd_eficiencia", "GCD: eficiencia culómbica por ciclo",
              "gcd", _panel_gcd_efficiency,
              "con la línea del 100 %, porque de qué lado cae es el "
              "resultado",
              technique="gcd"),
    PanelSpec("gcd_fuga", "GCD: la corriente parásita contra la aplicada",
              "gcd", _panel_gcd_leakage,
              "tiene que salir PLANA: es la misma fuga vista a ocho "
              "corrientes distintas",
              technique="gcd"),
    PanelSpec("gcd_resistencia", "GCD: la caída IR y la ESR", "gcd",
              _panel_gcd_resistance,
              "el salto sube con la corriente y la resistencia no, y eso "
              "es la comprobación de que el salto es óhmico",
              technique="gcd"),
    PanelSpec("gcd_ragone", "GCD: el Ragone con su techo", "gcd",
              _panel_ragone,
              "energía contra potencia en log-log, con el techo "
              "ΔV²/4R que ninguna descarga medida alcanza",
              technique="gcd"),
    PanelSpec("her_polarizacion", "HER/OER: la curva de polarización",
              "catalysis", _panel_her_polarisation,
              "η contra |j| sin logaritmo, que es donde se ve una "
              "limitación por transporte o un problema de burbujas",
              technique="catalysis"),
    PanelSpec("her_tafel", "HER/OER: el gráfico de Tafel", "catalysis",
              _panel_her_tafel,
              "con la REGIÓN sobre la que se ajustó la pendiente "
              "sombreada, para poder juzgar si llega a una década",
              technique="catalysis"),
)

PANELS_BY_KEY: dict[str, PanelSpec] = {p.key: p for p in ECHEM_PANELS}

#: The panels a figure starts with, by what is loaded. These are the
#: defaults because they are the figures the field publishes, not
#: because they are the first in the list: four CV panels that between
#: them say what was measured, what it gives, what leaks and what the
#: rate study resolves; and for impedance the Nyquist-with-inset and the
#: Bode, which is the pair every supercapacitor paper carries.
DEFAULT_PANELS: dict[str, tuple[str, ...]] = {
    "rates": ("cv_serie", "cv_capacitancia", "cv_carga", "cv_terminos"),
    "cv": ("cv_dunn",),
    "gcd": ("gcd_series", "gcd_velocidad", "gcd_ciclado", "gcd_ragone"),
    "eis": ("eis_nyquist", "eis_bode", "eis_capacitancia", "eis_residuo"),
    "ms": ("mott_schottky",),
    "catalysis": ("her_polarizacion", "her_tafel"),
}

#: The techniques the panel list can be filtered by, with ``"todo"``
#: first because that is the old behaviour and the one that has to stay
#: reachable in one click.
TECHNIQUES: tuple[tuple[str, str], ...] = (
    ("todo", "Todo"),
    ("cv", "CV (voltamperometría)"),
    ("gcd", "GCD (carga-descarga)"),
    ("eis", "EIS (impedancia)"),
    ("catalysis", "HER/OER (catálisis)"),
    ("ms", "Mott-Schottky"),
)


def available_panels(session, technique: str = "todo") -> list[PanelSpec]:
    """The panels whose data are loaded, optionally one technique only.

    A panel without data is left OUT of the list rather than offered and
    then drawn empty: an empty panel in a six-panel figure reads as a
    bug, and there is no way to tell it from one whose data were there
    and said nothing.

    The technique filter is on top of that and is a different question —
    not "could this be drawn" but "is this what I am drawing now". With
    twenty-four panels across five techniques in one list, the list is
    itself the obstacle.
    """
    have = _what_is_loaded(session)
    return [panel for panel in ECHEM_PANELS
            if have.get(panel.needs, False)
            and technique in ("todo", "", panel.technique)]


def techniques_with_data(session) -> list[tuple[str, str]]:
    """``(key, label)`` for ``"todo"`` plus every technique that has
    something loaded, so the selector cannot offer an empty list."""
    have = _what_is_loaded(session)
    present = {panel.technique for panel in ECHEM_PANELS
               if have.get(panel.needs, False)}
    return [(key, label) for key, label in TECHNIQUES
            if key == "todo" or key in present]


def _what_is_loaded(session) -> dict[str, bool]:
    return {
        "rates": len(session.rate_series) >= 2,
        "cv": session.cv is not None or bool(session.rate_series),
        "gcd": bool(getattr(session, "gcd_series", None)) or (
            getattr(session, "gcd", None) is not None),
        "eis": session.eis is not None or bool(session.eis_series),
        "ms": len(session.ms_series) >= 2,
        "catalysis": getattr(session, "lsv", None) is not None,
    }


def default_panels(session, technique: str = "todo") -> list[str]:
    """A figure's worth of panels for whatever is loaded.

    So that the common case is one click rather than a list of
    checkboxes: the person who just loaded a rate series wants the four
    panels that series makes, and choosing them by hand every time is
    the kind of friction that stops a figure being made at all.
    """
    have = _what_is_loaded(session)
    offered = {panel.key for panel in available_panels(session, technique)}
    chosen: list[str] = []
    for need in ("rates", "gcd", "eis", "catalysis", "ms"):
        if not have.get(need):
            continue
        chosen.extend(key for key in DEFAULT_PANELS[need] if key in offered)
    if not chosen and have.get("cv"):
        chosen.extend(key for key in DEFAULT_PANELS["cv"] if key in offered)
    return chosen


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

        placeholder(figure.add_subplot(111), palette, spec.text("pick_panel"))
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
            axis.text(0.5, 0.5, spec.text("failed", label=panel.label),
                      transform=axis.transAxes, ha="center", va="center",
                      fontsize=spec.font_size, color=palette.danger)
            continue
        # The scale override, applied after the panel drew: it is the
        # panel that knows what its data are, and this is the user
        # overruling it. Only the axis the panel was handed, never the
        # twins -- a Bode's phase twin runs -90 to 0 and a logarithmic
        # scale on it draws nothing.
        apply_scale(axis, spec.scale_for(key))
        if spec.letters:
            letter = f"({LETTERS[index % len(LETTERS)]})"
            if spec.letters_inside:
                axis.text(0.98, 0.04, letter, transform=axis.transAxes,
                          ha="right", va="bottom",
                          fontsize=spec.font_size + 0.5, fontweight="bold")
            else:
                axis.text(-0.02, 1.03, letter, transform=axis.transAxes,
                          ha="left", va="bottom",
                          fontsize=spec.font_size + 0.5, fontweight="bold")
    # The grid, settled once over every axis the figure ended up with.
    # Each panel sets it on the axis it was handed, and that leaves the
    # twin of a two-scale panel carrying the on-screen style's grid into
    # the exported figure, where nobody asked for one.
    for axes in every_axes(figure):
        _set_grid(axes, spec)
    if spec.title:
        figure.suptitle(spec.title, fontsize=spec.font_size + 1.5)
    figure.tight_layout(pad=1.0,
                        rect=(0, 0, 1, 0.96 if spec.title else 1))
    return problems


def save_composite(spec: CompositeFigure, session, palette: Palette,
                   path) -> tuple[Any, list[str]]:
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
    "every_axes",
    "DEFAULT_PANELS",
    "ECHEM_PANELS",
    "HER_V_VS_RHE",
    "LABELS",
    "LETTERS",
    "OER_V_VS_RHE",
    "PANELS_BY_KEY",
    "PRESETS",
    "PRESET_LABELS",
    "PanelSpec",
    "SERIES_COLOURS",
    "apply_preset",
    "available_panels",
    "default_panels",
    "draw_composite",
    "save_composite",
]
