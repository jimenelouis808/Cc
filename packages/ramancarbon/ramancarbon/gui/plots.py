"""Figures. Written against a matplotlib ``Axes``, never against a widget.

Keeping the drawing free of Tk means the same code produces the on-screen
plot and the PNG that goes into a report, and it can be tested with the Agg
backend on a machine with no display.

The colour code is fixed across the whole application (see
:mod:`ramancarbon.gui.theme`): data in neutral grey, the total fit in
orange, individual components in the categorical sequence, the residual in
muted grey below the axis, the baseline in the accent colour. A user learns
it once.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..analysis.report import AnalysisResult
from ..core.peaks import PeakMeasurement
from ..core.spectrum import Spectrum
from ..models.fitting import FitResult
from .theme import Palette


def _stagger(
    positions: Sequence[float],
    span: tuple[float, float],
    spacing: float = 0.04,
    base: float = 8.0,
    step: float = 11.0,
    levels: int = 3,
) -> list[float]:
    """Vertical offsets that keep close-together annotations from colliding.

    Labels are placed on the lowest level whose previous occupant is further
    away than ``spacing`` of the axis width. Nanotube spectra routinely show
    four RBM peaks inside 150 cm⁻¹, and without this their labels overprint
    each other into an unreadable smudge.

    Parameters
    ----------
    positions:
        Annotation abscissae, in data units.
    span:
        ``(low, high)`` of the axis, to make ``spacing`` a fraction of width.
    spacing:
        Minimum separation, as a fraction of the axis width, for two labels
        to share a level.
    base, step:
        First level's offset and the gap between levels, in points.
    levels:
        How many levels to cycle through.

    Returns
    -------
    list[float]
        One offset in points per position, in the given order.
    """
    width = max(span[1] - span[0], 1e-9)
    minimum = spacing * width
    last: list[float] = [-float("inf")] * levels
    offsets: list[float] = []
    for position in positions:
        chosen = levels - 1
        for level in range(levels):
            if position - last[level] >= minimum:
                chosen = level
                break
        last[chosen] = position
        offsets.append(base + chosen * step)
    return offsets


def plot_spectrum(
    ax,
    spectrum: Spectrum,
    palette: Palette,
    raw: Optional[Spectrum] = None,
    baseline: Optional[np.ndarray] = None,
    baseline_x: Optional[np.ndarray] = None,
    peaks: Optional[Sequence[PeakMeasurement]] = None,
    label_peaks: bool = True,
    title: Optional[str] = None,
) -> None:
    """Draw a spectrum, optionally with the raw trace and the baseline.

    Parameters
    ----------
    ax:
        Target axes.
    spectrum:
        The (processed) spectrum to show.
    palette:
        Colour scheme.
    raw:
        The unprocessed spectrum. Drawn faintly behind, so the user can see
        what preprocessing did rather than trusting that it was right.
    baseline, baseline_x:
        The subtracted background, drawn over the raw trace.
    peaks:
        Detected peaks to mark.
    label_peaks:
        Whether to annotate each marker with its position.
    title:
        Axes title.
    """
    if raw is not None:
        ax.plot(
            raw.shift,
            raw.intensity,
            color=palette.text_muted,
            alpha=0.35,
            linewidth=0.9,
            label="sin procesar",
        )
    if baseline is not None and baseline_x is not None and len(baseline) == len(baseline_x):
        ax.plot(
            baseline_x,
            baseline,
            color=palette.baseline,
            linestyle="--",
            linewidth=1.0,
            label="línea base",
        )
    ax.plot(
        spectrum.shift,
        spectrum.intensity,
        color=palette.data,
        linewidth=1.2,
        label=spectrum.name,
    )
    if peaks:
        xs = [p.position for p in peaks]
        ys = [spectrum.interpolate_at([p.position])[0] for p in peaks]
        ax.plot(xs, ys, linestyle="none", marker="v", markersize=5,
                color=palette.accent, label="picos")
        if label_peaks:
            for x, y, dy in zip(xs, ys, _stagger(xs, spectrum.range, spacing=0.035)):
                ax.annotate(
                    f"{x:.0f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, dy),
                    ha="center",
                    fontsize=7,
                    color=palette.text_muted,
                )
            ax.margins(y=0.18)
    ax.set_xlabel("Desplazamiento Raman (cm⁻¹)")
    ax.set_ylabel("Intensidad (u.a.)")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")


def plot_fit(
    axes,
    fit: FitResult,
    palette: Palette,
    residual_axes=None,
    title: Optional[str] = None,
) -> None:
    """Draw a deconvolution: data, components, total, and the residual.

    The residual panel is not decoration. Structure left in the residual —
    a systematic S-shape under a band, a bump where no component sits — is
    the only honest way to see that a model is missing a component, and it
    shows things that R² cannot: an R² of 0.999 routinely hides a residual
    twenty times the noise under the D band.

    Parameters
    ----------
    axes:
        Main axes for the fit.
    fit:
        The result to draw.
    palette:
        Colour scheme.
    residual_axes:
        Separate axes for the residual. When ``None``, the residual is
        drawn on the main axes offset below zero.
    title:
        Axes title.
    """
    x = fit.x
    axes.plot(x, fit.y, color=palette.data, linewidth=1.1, label="datos")
    axes.plot(x, fit.fitted, color=palette.fitted, linewidth=1.4, label="ajuste total")

    background = fit.background
    for index, component in enumerate(fit.peaks):
        curve = component.curve(x) + background
        colour = palette.component_colour(index)
        axes.plot(x, curve, color=colour, linewidth=1.0, alpha=0.95)
        axes.fill_between(x, background, curve, color=colour, alpha=0.13)
        peak_y = component.peak_height + np.interp(component.peak_position, x, background)
        axes.annotate(
            component.name,
            (component.peak_position, peak_y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            color=colour,
            fontweight="bold",
        )
    if np.any(background != 0):
        axes.plot(x, background, color=palette.text_muted, linestyle=":",
                  linewidth=0.9, label="fondo del ajuste")

    axes.set_ylabel("Intensidad (u.a.)")
    axes.legend(loc="upper right")
    if title:
        axes.set_title(title)

    noise = float(np.std(fit.residual)) or 1.0
    if residual_axes is not None:
        residual_axes.axhline(0.0, color=palette.border, linewidth=0.8)
        residual_axes.plot(x, fit.residual, color=palette.residual, linewidth=0.9)
        residual_axes.fill_between(x, 0, fit.residual, color=palette.residual, alpha=0.25)
        residual_axes.set_xlabel("Desplazamiento Raman (cm⁻¹)")
        residual_axes.set_ylabel("Residuo")
        limit = max(3.5 * noise, float(np.max(np.abs(fit.residual))) * 1.1, 1e-9)
        residual_axes.set_ylim(-limit, limit)
    else:
        axes.set_xlabel("Desplazamiento Raman (cm⁻¹)")


def plot_rbm(ax, result: AnalysisResult, palette: Palette) -> None:
    """The RBM region with each peak annotated by its deduced diameter."""
    spectrum = result.processed
    low = max(80.0, spectrum.range[0])
    high = min(400.0, spectrum.range[1])
    if high - low < 20.0:
        ax.text(
            0.5,
            0.5,
            "El espectro no cubre la región RBM\n(80–400 cm⁻¹)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=palette.text_muted,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    x, y = spectrum.region(low, high)
    ax.plot(x, y, color=palette.data, linewidth=1.2)
    if result.rbm.fit is not None:
        fit = result.rbm.fit
        ax.plot(fit.x, fit.fitted, color=palette.fitted, linewidth=1.2, alpha=0.9)
        for index, component in enumerate(fit.peaks):
            ax.plot(
                fit.x,
                component.curve(fit.x) + fit.background,
                color=palette.component_colour(index),
                linewidth=0.9,
            )
    # Just the diameter: the frequency is already on the x axis, and a
    # nanotube sample routinely shows three or four RBMs inside 150 cm-1.
    # A longer label on this small panel overprints its neighbour, and a
    # two-line one on three levels runs into the title.
    positions = [e.input_value for e in result.rbm.diameters]
    offsets = _stagger(positions, (low, high), spacing=0.07, base=7.0, step=12.0,
                       levels=2)
    for estimate, dy in zip(result.rbm.diameters, offsets):
        peak = spectrum.max_in(estimate.input_value - 6, estimate.input_value + 6)
        if peak is None:
            continue
        ax.annotate(
            f"{estimate.diameter_nm:.2f} nm",
            (estimate.input_value, peak[1]),
            textcoords="offset points",
            xytext=(0, dy),
            ha="center",
            fontsize=7.5,
            color=palette.text,
        )
    if offsets:
        # Headroom scaled to the highest label actually placed.
        ax.margins(y=0.14 + 0.012 * max(offsets))
    ax.set_xlabel("Desplazamiento Raman (cm⁻¹)")
    ax.set_ylabel("Intensidad (u.a.)")
    ax.set_title("Región RBM y diámetros")


def plot_strain_doping(ax, result: AnalysisResult, palette: Palette) -> None:
    """The (ω_G, ω_2D) plane with the strain and doping axes drawn on it.

    Seeing the measurement as a point relative to two lines is far clearer
    than reading two numbers: it shows at a glance whether the sample moved
    along the strain direction, the doping direction, or between them.
    """
    if result.shifts is None or result.shifts.decomposition is None:
        ax.text(
            0.5,
            0.5,
            "Hacen falta las bandas G y 2D\npara separar deformación y dopado",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=palette.text_muted,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    decomposition = result.shifts.decomposition
    origin_g = result.shifts.shifts.get("G") or result.shifts.shifts.get("G+")
    two_d = result.shifts.shifts.get("2D")
    g0 = origin_g.reference
    d0 = two_d.reference

    span = max(abs(decomposition.delta_g), abs(decomposition.delta_2d) / 2.2, 12.0) * 1.4
    t = np.linspace(-span, span, 2)
    ax.plot(g0 + t, d0 + 2.2 * t, color=palette.success, linewidth=1.2,
            label="deformación (pendiente 2.2)")
    ax.plot(g0 + t, d0 + 0.7 * t, color=palette.warning, linewidth=1.2,
            label="dopado (pendiente 0.7)")
    ax.plot([g0], [d0], marker="o", markersize=6, color=palette.text_muted,
            linestyle="none", label="referencia prístina")
    ax.plot([origin_g.measured], [two_d.measured], marker="*", markersize=14,
            color=palette.fitted, linestyle="none", label="muestra")
    ax.annotate(
        "",
        xy=(g0 + decomposition.strain_component_g,
            d0 + 2.2 * decomposition.strain_component_g),
        xytext=(g0, d0),
        arrowprops=dict(arrowstyle="->", color=palette.success, linewidth=1.4),
    )
    ax.annotate(
        "",
        xy=(origin_g.measured, two_d.measured),
        xytext=(g0 + decomposition.strain_component_g,
                d0 + 2.2 * decomposition.strain_component_g),
        arrowprops=dict(arrowstyle="->", color=palette.warning, linewidth=1.4),
    )
    ax.set_xlabel("ω(G) (cm⁻¹)")
    ax.set_ylabel("ω(2D) (cm⁻¹)")
    ax.set_title("Separación deformación / dopado")
    ax.legend(loc="best", fontsize=7.5)


def plot_overlay(ax, spectra: Sequence[Spectrum], palette: Palette,
                 offset: float = 0.0) -> None:
    """Overlay several spectra, optionally stacked with a vertical offset."""
    for index, spectrum in enumerate(spectra):
        y = spectrum.intensity
        if offset:
            scale = float(np.max(y) - np.min(y)) or 1.0
            y = y + index * offset * scale
        ax.plot(spectrum.shift, y, color=palette.component_colour(index),
                linewidth=1.1, label=spectrum.name)
    ax.set_xlabel("Desplazamiento Raman (cm⁻¹)")
    ax.set_ylabel("Intensidad (u.a.)")
    if len(spectra) <= 8:
        ax.legend(loc="upper right", fontsize=7.5)


def figure_for_report(result: AnalysisResult, palette: Palette, figsize=(9.0, 7.0)):
    """A four-panel summary figure suitable for export.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.figure import Figure

    figure = Figure(figsize=figsize)
    grid = figure.add_gridspec(3, 2, height_ratios=(2.2, 1.0, 2.0), hspace=0.45, wspace=0.28)

    overview = figure.add_subplot(grid[0, :])
    plot_spectrum(
        overview,
        result.processed,
        palette,
        peaks=result.peaks,
        title=f"{result.raw.name} — {result.classification.label}",
    )

    if result.fit is not None:
        fit_ax = figure.add_subplot(grid[1:3, 0])
        plot_fit(fit_ax, result.fit, palette, title="Deconvolución D–G")
    rbm_ax = figure.add_subplot(grid[1, 1])
    plot_rbm(rbm_ax, result, palette)
    shift_ax = figure.add_subplot(grid[2, 1])
    plot_strain_doping(shift_ax, result, palette)
    figure.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.08)
    return figure


__all__ = [
    "figure_for_report",
    "plot_fit",
    "plot_overlay",
    "plot_rbm",
    "plot_spectrum",
    "plot_strain_doping",
]
