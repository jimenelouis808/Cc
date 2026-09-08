"""Diffraction figures. Written against matplotlib axes, never against Tk.

The Rietveld plot follows the convention the field settled on decades ago
and it is worth stating why, because it is not arbitrary: observed points,
calculated line, tick marks per phase, and the difference curve *below* the
axis on the same intensity scale. Everything about that layout exists to
make the difference curve easy to read — it is the part that says where the
model fails, and an R factor cannot. Putting the difference on its own
rescaled axis, which looks tidier, destroys exactly that comparison.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..xrd.pattern import Pattern
from ..xrd.powder import reflections
from ..xrd.report import XRDResult
from ..xrd.rietveld import RietveldResult
from ..xrd.search import XRDPeak
from .theme import Palette

#: Vertical space, as a fraction of the data range, reserved for the tick
#: rows and the difference curve below the pattern.
TICK_BAND = 0.16
DIFFERENCE_BAND = 0.22


def plot_pattern(
    ax,
    pattern: Pattern,
    palette: Palette,
    peaks: Optional[Sequence[XRDPeak]] = None,
    unexplained: Optional[Sequence[XRDPeak]] = None,
    log_scale: bool = False,
) -> None:
    """The measured diffractogram, with the peaks that were found."""
    ax.plot(
        pattern.two_theta,
        pattern.intensity,
        color=palette.data,
        linewidth=0.9,
        label="medido",
    )
    unexplained_ids = {id(p) for p in unexplained or ()}
    if peaks:
        top = float(pattern.intensity.max())
        for peak in peaks:
            odd = id(peak) in unexplained_ids
            ax.axvline(
                peak.two_theta,
                color=palette.danger if odd else palette.accent,
                linewidth=1.2 if odd else 0.6,
                alpha=0.75 if odd else 0.35,
            )
            if odd:
                ax.annotate(
                    f"{peak.two_theta:.2f}°",
                    xy=(peak.two_theta, min(peak.height * 1.5, top)),
                    fontsize=7,
                    color=palette.danger,
                    rotation=90,
                    ha="center",
                    va="bottom",
                )
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("2θ (grados)")
    ax.set_ylabel("Intensidad (cuentas)")
    ax.set_xlim(*pattern.range)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def plot_rietveld(
    ax,
    refinement: RietveldResult,
    palette: Palette,
    show_phases: bool = True,
) -> None:
    """Observed, calculated, reflection ticks and the difference below."""
    pattern = refinement.pattern
    observed = pattern.intensity
    calculated = refinement.calculated
    difference = refinement.difference

    top = float(max(observed.max(), calculated.max()))
    bottom = float(min(observed.min(), 0.0))
    span = max(top - bottom, 1e-9)

    ax.plot(
        pattern.two_theta, observed, "o", markersize=1.6,
        color=palette.data, label="observado", zorder=2,
    )
    ax.plot(
        pattern.two_theta, calculated, color=palette.fitted,
        linewidth=1.1, label="calculado", zorder=3,
    )
    ax.plot(
        pattern.two_theta, refinement.background, color=palette.baseline,
        linewidth=0.8, linestyle="--", alpha=0.7, label="fondo", zorder=1,
    )

    rows = len(refinement.phases) if show_phases else 0
    tick_top = bottom - 0.02 * span
    row_height = TICK_BAND * span / max(rows, 1)
    for index, phase in enumerate(refinement.phases):
        if not show_phases:
            break
        crystal = phase.current_crystal()
        colour = palette.component_colour(index)
        y = tick_top - index * row_height
        lines = reflections(
            crystal,
            wavelength=pattern.wavelength,
            two_theta_range=pattern.range,
        )
        angles = [r.two_theta + refinement.zero for r in lines]
        ax.plot(
            angles, [y] * len(angles), "|", color=colour, markersize=6,
            markeredgewidth=1.0, label=crystal.name, zorder=4,
        )

    offset = tick_top - rows * row_height - DIFFERENCE_BAND * span * 0.35
    ax.plot(
        pattern.two_theta, difference + offset, color=palette.residual,
        linewidth=0.8, label="diferencia", zorder=2,
    )
    ax.axhline(offset, color=palette.border, linewidth=0.6, zorder=1)

    ax.set_xlabel("2θ (grados)")
    ax.set_ylabel("Intensidad (cuentas)")
    ax.set_xlim(*pattern.range)
    ax.set_ylim(offset - 0.5 * DIFFERENCE_BAND * span, top * 1.06)
    ax.legend(loc="upper right", frameon=False, fontsize=7, ncol=2)
    ax.set_title(
        f"Rwp = {100 * refinement.r_wp:.2f} %   GOF = {refinement.gof:.3f}",
        fontsize=9,
    )


def plot_phase_sticks(
    ax, result: XRDResult, palette: Palette, top_n: int = 3
) -> None:
    """Calculated stick patterns of the identified phases against the data.

    Sticks rather than profiles, because at this stage the question is
    "are the lines in the right places", and a profile invites the eye to
    compare intensities that preferred orientation has already changed.
    """
    pattern = result.pattern
    ax.plot(
        pattern.two_theta,
        pattern.intensity / max(float(pattern.intensity.max()), 1e-9) * 100.0,
        color=palette.data,
        linewidth=0.8,
        label="medido (normalizado)",
    )
    for index, match in enumerate(result.search.accepted[:top_n]):
        colour = palette.component_colour(index)
        lines = reflections(
            match.crystal, wavelength=pattern.wavelength,
            two_theta_range=pattern.range,
        )
        for line in lines:
            ax.vlines(
                line.two_theta + match.zero_shift,
                0.0,
                -line.intensity * 0.6 - index * 65.0,
                color=colour,
                linewidth=1.0,
            )
        ax.plot([], [], color=colour, label=match.crystal.name)
    ax.axhline(0.0, color=palette.border, linewidth=0.6)
    ax.set_xlabel("2θ (grados)")
    ax.set_ylabel("Intensidad relativa")
    ax.set_xlim(*pattern.range)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def plot_weighted_difference(ax, refinement: RietveldResult, palette: Palette) -> None:
    """The difference divided by the uncertainty, which is what χ² sums.

    Reading the plain difference underweights the low-angle end, where the
    counts and therefore the uncertainties are largest. In σ units every
    point is on the same footing, and a systematic excursion of three or
    four σ over a range of angles is visible where the same feature is
    invisible in counts.
    """
    pattern = refinement.pattern
    weighted = refinement.difference / np.maximum(pattern.sigma, 1e-9)
    ax.plot(pattern.two_theta, weighted, color=palette.residual, linewidth=0.7)
    for level, style in ((3.0, ":"), (-3.0, ":")):
        ax.axhline(level, color=palette.warning, linewidth=0.7, linestyle=style)
    ax.axhline(0.0, color=palette.border, linewidth=0.6)
    ax.set_xlabel("2θ (grados)")
    ax.set_ylabel("(obs − calc) / σ")
    ax.set_xlim(*pattern.range)
    ax.set_title("Diferencia en unidades de σ (las líneas son ±3σ)", fontsize=9)


def figure_for_report(refinement: RietveldResult, palette: Palette,
                      figsize=(9.0, 7.0)):
    """A two-panel figure for saving: Rietveld plot and weighted difference."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    figure = Figure(figsize=figsize, dpi=150)
    figure.patch.set_facecolor(palette.surface)
    top = figure.add_subplot(2, 1, 1)
    bottom = figure.add_subplot(2, 1, 2, sharex=top)
    for axis in (top, bottom):
        axis.set_facecolor(palette.surface)
        for spine in axis.spines.values():
            spine.set_color(palette.border)
        axis.tick_params(colors=palette.text_muted, labelsize=8)
        axis.xaxis.label.set_color(palette.text)
        axis.yaxis.label.set_color(palette.text)
        axis.title.set_color(palette.text)
    plot_rietveld(top, refinement, palette)
    plot_weighted_difference(bottom, refinement, palette)
    figure.tight_layout()
    return figure


__all__ = [
    "DIFFERENCE_BAND",
    "TICK_BAND",
    "figure_for_report",
    "plot_pattern",
    "plot_phase_sticks",
    "plot_rietveld",
    "plot_weighted_difference",
]
