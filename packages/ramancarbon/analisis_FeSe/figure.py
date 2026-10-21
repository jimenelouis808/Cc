"""The Rietveld figure, drawn from the saved columns.

House convention, which is also the journal convention and is in
CLAUDE.md: observed as points, calculated as a line over them, the
reflection ticks in one row per phase, and the difference BELOW on the
SAME vertical scale -- a difference curve on its own rescaled axis looks
cleaner and destroys the one comparison it is for.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

OUT = "/home/user/Cc/packages/ramancarbon/analisis_FeSe"

# Validated against the dataviz six checks (light surface, categorical):
# lightness band, chroma floor, CVD separation and normal-vision floor all
# pass; the contrast WARN is relieved by labelling every tick row directly,
# so identity never rests on colour alone.
PHASE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
OBSERVED = "#1a1a19"
CALCULATED = "#d4213d"
BACKGROUND_LINE = "#8a8a84"
RESIDUAL = "#52514e"

TITLES = {
    "S1_900": "Muestra S1 — 900 °C",
    "S2_1020": "Muestra S2 — 1020 °C",
}


def draw(sample, axis, show_xlabel=True):
    table = np.loadtxt(f"{OUT}/{sample}_ajuste.csv", delimiter=",", skiprows=1)
    two_theta, observed, calculated, background, difference = table[:, :5].T
    with open(f"{OUT}/{sample}_resumen.json") as handle:
        summary = json.load(handle)
    ticks = np.load(f"{OUT}/{sample}_ticks.npz")
    names = [row["phase"] for row in summary["phases"]]
    pretty = {row["phase"]: row["pretty"] for row in summary["phases"]}

    top = float(max(observed.max(), calculated.max()))
    floor = float(min(observed.min(), background.min()))
    span = top - floor

    # Observed as points and calculated as a line, not the other way
    # round: the data are the measurement and the model is the claim, and
    # a line drawn through points is read as a fit while two lines are
    # read as two measurements.
    axis.plot(two_theta, observed, ".", color=OBSERVED, markersize=1.0,
              alpha=0.55, label="observado", zorder=2, rasterized=True)
    axis.plot(two_theta, calculated, "-", color=CALCULATED, linewidth=0.9,
              label="calculado", zorder=3)
    axis.plot(two_theta, background, "--", color=BACKGROUND_LINE,
              linewidth=0.7, label="fondo", zorder=2)

    rows = len(names)
    tick_top = floor - 0.06 * span
    row_height = 0.045 * span
    handles = []
    for index, name in enumerate(names):
        y = tick_top - index * row_height
        angles = ticks[name]
        colour = PHASE_COLOURS[index % len(PHASE_COLOURS)]
        if angles.size:
            axis.plot(angles, np.full(angles.shape, y), "|", color=colour,
                      markersize=4.5, markeredgewidth=0.8, zorder=2)
        # A second legend keyed to the tick rows, rather than names
        # written beside them: those needed eleven degrees of empty axis
        # to fit into, which drew a bottom spine and tick marks out to
        # 100 deg where nothing was measured. Identity still never rests
        # on colour alone, which is what the contrast check asks for.
        handles.append(Line2D([], [], color=colour, marker="|", linestyle="",
                              markersize=6, markeredgewidth=1.1,
                              label=pretty.get(name, name)))

    offset = tick_top - rows * row_height - 0.07 * span
    axis.plot(two_theta, difference + offset, "-", color=RESIDUAL,
              linewidth=0.6, label="diferencia", zorder=3)
    axis.axhline(offset, color="#c9c9c2", linewidth=0.5, zorder=1)
    # Below the baseline, not above it: at the left edge the residual
    # itself occupies the band the label was sitting in, and a label
    # drawn over the curve it names is worse than no label.
    axis.text(two_theta[0] + 0.8, offset - 0.105 * span, "obs − calc",
              color=RESIDUAL, fontsize=6, va="center", ha="left")

    axis.set_xlim(two_theta[0], two_theta[-1])
    axis.set_ylim(offset - 0.14 * span, top + 0.07 * span)
    axis.set_ylabel("Intensidad (cuentas)", fontsize=7)
    if show_xlabel:
        axis.set_xlabel("2θ (grados, Cu Kα₁)", fontsize=7)
    axis.tick_params(labelsize=6.5, width=0.6, length=2.6)
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.tick_params(which="minor", width=0.4, length=1.4)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_linewidth(0.6)

    axis.set_title(TITLES.get(sample, sample), fontsize=8, loc="left",
                   pad=3)
    axis.text(
        0.995, 0.96,
        "$R_{wp}$ = %.2f %%   $R_p$ = %.2f %%   GOF = %.3f"
        % (100 * summary["r_wp"], 100 * summary["r_p"], summary["gof"]),
        transform=axis.transAxes, fontsize=6.2, ha="right", va="top",
        color="#52514e")
    curves = axis.legend(fontsize=6, frameon=False, loc="upper right",
                         bbox_to_anchor=(0.995, 0.90), handlelength=1.6,
                         labelspacing=0.25)
    axis.add_artist(curves)
    axis.legend(handles=handles, fontsize=6, frameon=False, loc="upper right",
                bbox_to_anchor=(0.995, 0.56), handlelength=1.0,
                labelspacing=0.32, handletextpad=0.5)
    return summary


def main(samples):
    plt.rcParams["font.family"] = "DejaVu Sans"
    height = 3.0 * len(samples)
    figure, axes = plt.subplots(len(samples), 1, figsize=(7.0, height))
    if len(samples) == 1:
        axes = [axes]
    for index, (sample, axis) in enumerate(zip(samples, axes)):
        draw(sample, axis, show_xlabel=(index == len(samples) - 1))
    figure.tight_layout(h_pad=1.6)
    for extension, dpi in (("png", 600), ("pdf", None)):
        figure.savefig(f"{OUT}/rietveld_FeSe.{extension}",
                       dpi=dpi, bbox_inches="tight")
    print("escrito", f"{OUT}/rietveld_FeSe.png y .pdf")


if __name__ == "__main__":
    main(sys.argv[1:] or ["S1_900", "S2_1020"])
