"""XPS figures. Written against matplotlib axes, never Tk.

One convention is followed throughout because the whole field follows it,
and a plot that departs from it is read backwards by everybody who looks at
it: **the binding-energy axis increases to the left.** That is a drawing
convention, not a property of the data — the arrays are stored ascending —
and it exists because the spectra were originally recorded against kinetic
energy, which does increase to the right.

The region plot draws the residual underneath, on the same scale, for the
same reason the Rietveld plot on the diffraction side does: a residual on
its own rescaled axis always looks like noise, and the comparison the panel
exists for is destroyed by making it pretty.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..xps.fitting import XPSFitResult
from ..xps.quantify import Quantification
from ..xps.spectrum import XPSSpectrum
from ..xps.survey import SurveyResult
from .theme import Palette


def _invert(ax) -> None:
    """Binding energy increases to the left, as everybody draws it."""
    if not ax.xaxis_inverted():
        ax.invert_xaxis()


def plot_spectrum(ax, spectrum: XPSSpectrum, palette: Palette,
                  label: Optional[str] = None) -> None:
    """One spectrum, raw."""
    ax.plot(spectrum.binding_energy, spectrum.counts, color=palette.data,
            linewidth=0.9, label=label or spectrum.name)
    ax.set_xlabel("Energía de enlace (eV)")
    ax.set_ylabel(f"Intensidad ({spectrum.intensity_unit})")
    _invert(ax)


def plot_survey(ax, spectrum: XPSSpectrum, palette: Palette,
                result: Optional[SurveyResult] = None,
                max_labels: int = 14) -> None:
    """A survey with its identified lines labelled.

    Only the strongest line of each element is labelled, and at most
    ``max_labels`` of them. A survey has thirty or forty peaks and labelling
    all of them produces a wall of overlapping text that hides the spectrum
    it is annotating — which is the opposite of what the labels are for.
    """
    ax.plot(spectrum.binding_energy, spectrum.counts, color=palette.data,
            linewidth=0.8)
    if result is not None:
        best: list[tuple[float, str, float]] = []
        for element in result.elements:
            match = element.primary_match or (element.matched[0]
                                              if element.matched else None)
            if match is None:
                continue
            best.append((match.height, f"{element.symbol} {match.label.split()[-1]}",
                         match.observed_ev))
        best.sort(reverse=True)
        for height, text, energy in best[:max_labels]:
            ax.annotate(
                text, xy=(energy, height), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=7,
                color=palette.fitted, rotation=90,
            )
        for peak in result.unexplained[:6]:
            ax.plot([peak.binding_energy], [peak.height], "v",
                    color=palette.danger, markersize=5)
    ax.set_xlabel("Energía de enlace (eV)")
    ax.set_ylabel(f"Intensidad ({spectrum.intensity_unit})")
    ax.margins(y=0.18)
    # Counts do not go below zero, and the headroom the rotated labels need
    # is all at the top.
    ax.set_ylim(bottom=0.0)
    _invert(ax)


def plot_region(figure, result: XPSFitResult, palette: Palette,
                show_residual: bool = True) -> None:
    """A fitted region: data, background, components, envelope, residual.

    The residual goes underneath **on the same intensity scale**. Putting it
    on its own rescaled axis looks tidier and destroys the only comparison
    the panel is for: whether what the model failed to describe is large
    compared with the peaks, or small.
    """
    if show_residual:
        # subplots(), not add_gridspec() + add_subplot(): a gridspec that
        # carries its own hspace makes axes tight_layout refuses to place,
        # and every canvas in the suite is laid out with tight_layout.
        ax, below = figure.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": (3.2, 1.0)}
        )
    else:
        ax = figure.add_subplot(1, 1, 1)
        below = None

    energy = result.energy
    ax.plot(energy, result.counts, "o", markersize=2.0, color=palette.data,
            markerfacecolor="none", markeredgewidth=0.6, label="datos")
    ax.plot(energy, result.background.values, color=palette.baseline,
            linewidth=1.0, linestyle="--", label=result.background.kind)
    for index, component in enumerate(result.components):
        curve = component.curve(energy) + result.background.values
        colour = palette.component_colour(index)
        ax.fill_between(energy, result.background.values, curve, color=colour,
                        alpha=0.25, linewidth=0)
        ax.plot(energy, curve, color=colour, linewidth=0.9,
                label=component.label)
    ax.plot(energy, result.fitted, color=palette.fitted, linewidth=1.2,
            label="envolvente")
    ax.set_ylabel("Intensidad")
    ax.legend(loc="upper left", frameon=False, fontsize=7, ncol=2)
    ax.set_title(
        f"{result.region_label}   χ²_red = {result.reduced_chi2:.2f}"
        + (f"   DW = {result.durbin_watson:.2f}" if result.durbin_watson else ""),
        fontsize=9,
    )
    _invert(ax)

    if below is not None:
        below.axhline(0.0, color=palette.border, linewidth=0.6)
        below.plot(energy, result.residual, color=palette.residual, linewidth=0.8)
        below.set_ylabel("Residuo")
        below.set_xlabel("Energía de enlace (eV)")
        span = float(np.max(np.abs(result.counts - result.background.values)))
        if span > 0:
            below.set_ylim(-0.15 * span, 0.15 * span)
        _invert(below)
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("Energía de enlace (eV)")


def plot_composition(ax, composition: Quantification, palette: Palette) -> None:
    """Atomic per cent as a bar chart, largest first."""
    elements = [item.element for item in composition.abundances]
    values = [item.atomic_percent for item in composition.abundances]
    colours = [palette.component_colour(index) for index in range(len(values))]
    positions = np.arange(len(values))
    ax.bar(positions, values, color=colours, width=0.62)
    for position, value in zip(positions, values):
        ax.annotate(f"{value:.1f} %", xy=(position, value), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=palette.text)
    ax.set_xticks(positions)
    ax.set_xticklabels(elements)
    ax.set_ylabel("% atómico de lo detectado")
    ax.margins(y=0.16)


def plot_states(ax, results: Sequence[XPSFitResult], palette: Palette) -> None:
    """Stacked bars: the chemical states of each fitted region.

    One bar per region, split by component. It is the picture of the result
    a paper actually shows for a doped carbon — how the nitrogen divides
    between pyridinic, pyrrolic and graphitic — and it is a share of that
    region, never of the sample.
    """
    labels = [result.region_label for result in results]
    positions = np.arange(len(results))
    bottoms = np.zeros(len(results))
    seen: dict[str, int] = {}
    for index, result in enumerate(results):
        for component in result.components:
            share = 100.0 * component.area_fraction
            colour_index = seen.setdefault(component.label, len(seen))
            ax.bar(positions[index], share, bottom=bottoms[index],
                   color=palette.component_colour(colour_index), width=0.6,
                   label=component.label if index == 0 or
                   component.label not in seen else None)
            if share > 7.0:
                ax.annotate(
                    component.label.split()[0], fontsize=6.5,
                    xy=(positions[index], bottoms[index] + share / 2),
                    ha="center", va="center", color=palette.accent_text,
                )
            bottoms[index] += share
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("% del área de la región")
    ax.set_ylim(0, 105)


def plot_count_comparison(ax, comparisons: Sequence, palette: Palette) -> None:
    """χ² against the number of components, with Durbin–Watson beside it.

    Both, because they answer different halves of the question. χ² falling
    towards 1 says the model has become adequate; a Durbin–Watson climbing
    towards 2 says what is left is noise rather than structure. A χ² that
    keeps falling while DW is already at 2 is a model fitting noise, and
    the extra component it bought is not a chemical state.

    DW is annotated rather than drawn on a second y-axis: a twin axis
    cannot be laid out automatically alongside the rest of the section's
    figures, and two lines on incommensurate scales invite reading a
    crossing that means nothing.
    """
    counts = [item.n for item in comparisons]
    chi2 = [item.reduced_chi2 for item in comparisons]
    ax.plot(counts, chi2, "o-", color=palette.data)
    ax.axhline(1.0, color=palette.border, linewidth=0.7, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlabel("número de componentes")
    ax.set_ylabel("χ²_red  (1 = dentro del ruido)")
    ax.set_xticks(counts)
    for item, value in zip(comparisons, chi2):
        watson = item.durbin_watson
        text = f"DW {watson:.2f}" if watson is not None else "DW —"
        if not item.significant:
            text += "\nalguna componente\nno llega al ruido"
        ax.annotate(text, xy=(item.n, value), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=6.5,
                    color=palette.text_muted)
    ax.margins(y=0.25)


def plot_complex_capacitance(ax, analysis, palette: Palette) -> None:
    """C′ and C″ against frequency — the supercapacitor's own plot.

    Kept here beside the XPS figures only because it shares the shape of
    the problem; the electrochemistry section imports it.
    """
    ax.semilogx(analysis.frequency, 1e3 * analysis.real, color=palette.data,
                label="C′")
    ax.semilogx(analysis.frequency, 1e3 * analysis.imaginary,
                color=palette.fitted, label="C″")
    if analysis.relaxation_frequency_hz:
        ax.axvline(analysis.relaxation_frequency_hz, color=palette.border,
                   linewidth=0.8, linestyle=":")
        ax.annotate(f"τ₀ = {analysis.relaxation_s:.3g} s",
                    xy=(analysis.relaxation_frequency_hz,
                        1e3 * float(np.max(analysis.imaginary))),
                    xytext=(4, 0), textcoords="offset points", fontsize=7,
                    color=palette.text_muted)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Capacitancia (mF)")
    ax.legend(loc="upper right", frameon=False, fontsize=7)


__all__ = [
    "plot_complex_capacitance",
    "plot_composition",
    "plot_count_comparison",
    "plot_region",
    "plot_spectrum",
    "plot_states",
    "plot_survey",
]
