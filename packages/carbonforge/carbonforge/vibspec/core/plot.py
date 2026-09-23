"""The comparison figure: computed IR against experimental FTIR.

Built on :class:`matplotlib.figure.Figure`, not pyplot, so it runs headless
on a cluster and can be embedded in a GUI later without pyplot managing (and
leaking) its figures. Drawing is separate from figure creation for the same
reason: :func:`draw_ir_comparison` takes any axes.

Design choices, each for a reason:

* **One y axis, both curves normalised to 1.** The calculation is in
  (D/Å)²/amu, the experiment in absorbance; two y scales would invite reading
  a meaning into their relative heights that neither has.
* **Wavenumber decreasing to the right**, the FTIR convention, so the figure
  reads like the spectrometer's output.
* **Two colours, validated** for colour-vision deficiency and contrast
  (blue and orange of the reference categorical palette), plus a legend and
  line-style difference so identity never rests on colour alone.
* **Sticks under the computed curve**, so a band made of several modes is
  visible as such.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ...results.spectra import VibrationalSpectrum
from .analysis import MID_IR, BandMatch, Profile, computed_curve, normalise, sticks

COMPUTED_COLOUR = "#2a78d6"
EXPERIMENT_COLOUR = "#eb6834"
_TEXT = "#3b3b38"
_MUTED = "#8a8980"
_GRID = "#e6e5df"


def draw_ir_comparison(
    ax,
    spectrum: VibrationalSpectrum,
    experiment: Optional[tuple[np.ndarray, np.ndarray]] = None,
    fwhm_cm1: float = 10.0,
    profile: Profile = "lorentzian",
    scale_factor: float = 1.0,
    window: tuple[float, float] = MID_IR,
    offset: float = 0.0,
    matches: Optional[Sequence[BandMatch]] = None,
    computed_label: str = "Calculado",
    experiment_label: str = "FTIR",
    n_points: int = 4000,
) -> dict[str, np.ndarray]:
    """Draw the computed spectrum, and optionally the experiment, on ``ax``.

    Parameters
    ----------
    ax
        Matplotlib axes.
    spectrum
        Computed modes, rigid-body modes already removed.
    experiment
        ``(wavenumber, normalised absorbance)`` from
        :func:`~carbonforge.vibspec.core.analysis.prepare_experiment`.
    fwhm_cm1, profile, scale_factor
        Broadening and frequency scaling of the computed lines.
    window
        Wavenumber range shown, cm^-1.
    offset
        Vertical shift of the computed curve above the experiment. 0 overlays
        them (best for comparing positions); ~1.1 stacks them (best for
        comparing shapes).
    matches
        From :func:`~carbonforge.vibspec.core.analysis.match_bands`: matched
        experimental bands get a faint marker line and their computed value.

    Returns
    -------
    dict
        The numbers actually drawn: ``grid``, ``computed`` (normalised),
        ``stick_positions``, ``stick_heights``.
    """
    low, high = min(window), max(window)
    grid = np.linspace(low, high, n_points)
    intensity = computed_curve(spectrum, grid, fwhm_cm1=fwhm_cm1, profile=profile,
                               scale_factor=scale_factor)
    computed = normalise(grid, intensity) if intensity.max() > 0 else intensity
    positions, heights = sticks(spectrum, scale_factor)
    visible = (positions >= low) & (positions <= high) & (heights > 0)

    ax.vlines(positions[visible], offset, offset + heights[visible], color=COMPUTED_COLOUR,
              linewidth=0.8, alpha=0.45)
    ax.plot(grid, computed + offset, color=COMPUTED_COLOUR, linewidth=1.6,
            label=f"{computed_label} (×{scale_factor:.3f}, FWHM {fwhm_cm1:g} cm⁻¹)")

    if experiment is not None:
        x, y = experiment
        ax.plot(x, y, color=EXPERIMENT_COLOUR, linewidth=1.4, linestyle=(0, (5, 1.5)),
                label=experiment_label)

    if matches:
        for match in matches:
            if match.experimental_cm1 is None or not low <= match.experimental_cm1 <= high:
                continue
            ax.axvline(match.experimental_cm1, color=_MUTED, linewidth=0.6,
                       linestyle=":", zorder=0)
        # Label the strongest matches, skipping any that would overlap one
        # already placed: a selective label beats a pile of unreadable ones.
        min_gap = 0.035 * (high - low)
        placed: list[float] = []
        strongest = sorted((m for m in matches if m.experimental_cm1 is not None),
                           key=lambda m: -m.relative_intensity)
        for match in strongest:
            position = match.experimental_cm1
            if not low <= position <= high or any(abs(position - p) < min_gap for p in placed):
                continue
            placed.append(position)
            ax.annotate(f"{match.computed_cm1:.0f}", (position, 1.02 + offset),
                        ha="center", va="bottom", fontsize=7, color=_TEXT)
            if len(placed) == 8:
                break

    ax.set_xlim(high, low)
    top = 1.15 + offset
    ax.set_ylim(-0.03, top)
    ax.set_xlabel("Número de onda (cm⁻¹)", color=_TEXT)
    ax.set_ylabel("Absorbancia normalizada (u.a.)", color=_TEXT)
    ax.grid(axis="x", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_TEXT, labelsize=8)
    # Outside the data area, top right: the high-wavenumber end on the left
    # is exactly where O-H, N-H and C-H bands sit.
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=2, labelcolor=_TEXT, borderaxespad=0.2)

    return {"grid": grid, "computed": computed,
            "stick_positions": positions[visible], "stick_heights": heights[visible]}


def plot_ir_comparison(spectrum: VibrationalSpectrum, title: Optional[str] = None, **kwargs):
    """A standalone :class:`~matplotlib.figure.Figure`; save it with ``fig.savefig``.

    Keyword arguments go to :func:`draw_ir_comparison`. Returns
    ``(figure, drawn)`` where ``drawn`` is that function's return value.
    """
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.0, 4.0), layout="constrained")
    ax = figure.add_subplot()
    drawn = draw_ir_comparison(ax, spectrum, **kwargs)
    if title:
        ax.set_title(title, fontsize=10, color=_TEXT, loc="left")
    return figure, drawn
