"""Excess uptake against pore width, at three pressures.

The figure exists to answer one question: **does any of the exotic
carbon buy anything that a plain graphene slit of the same pore width
does not already give?** So the slit series is drawn as a line -- it is
a controlled sweep, the one series here where width moves and nothing
else does -- and the six frameworks measured earlier are dropped on top
as points at their own measured widths.

Three panels rather than one, because the optimum width MOVES with
pressure, and a single panel would either hide that or need a second
axis. Small multiples share the axes, so the shift is read by looking
across.

The y axis is excess wt%, not uptake per unit area. The per-area ratio
is the fairer comparison when areas differ, and the README carries it;
but the slits win on the raw quantity too, so the figure can use the
one that needs no explaining.

The x axis is the LARGEST FREE SPHERE -- the widest sphere that can
cross the cell -- not the centre-to-centre separation of the slit's
walls, which is about a carbon diameter larger. Only the free width is
defined for all ten structures, so only the free width puts them on one
axis. It is logarithmic because the structures span 3 to 62 A.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Reference palette, slots 1-3. Validated for this chart form with
# `validate_palette.js --pairs all --mode light`: worst CVD dE 9.2,
# worst normal-vision dE 24.0. The aqua warns on contrast against the
# surface, which obliges visible direct labels -- it has them.
SLIT, SCHWARZITE, SUPER = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"
PRESSURES = (1.0, 3.8, 10.0)

HERE = Path(__file__).resolve().parent
SCHWARZITES = HERE.parent / "validacion_schwarzitas"
SUPERS = HERE.parent / "validacion_superestructuras"
#: Hand-set because the log locator's default here is 3x10^0, 4x10^0,
#: ... -- scientific notation on single-digit angstroms, and enough of
#: them to collide. These are the widths worth reading off.
X_TICKS = (3, 5, 7, 10, 20, 40, 60)
#: H2's Lennard-Jones diameter. The series' structure is at its
#: multiples: a channel of n*sigma is the narrowest that holds n layers
#: of hydrogen, so each one is where a layer enters and the uptake
#: jumps. The lines are drawn in muted ink because they are an
#: explanation laid over the data, not a measurement.
SIGMA_H2 = 2.958
LAYERS = (1, 2, 3)


def excess_weight_percent(point: dict, porosity: dict) -> float:
    """Excess as a percentage of total mass, from the excess mass."""
    volume_cm3 = porosity["cell_volume_a3"] * 1e-24
    excess = point["densidad_exceso_g_cm3"] * volume_cm3
    framework = porosity["framework_mass_g"]
    return 100.0 * excess / (excess + framework)


def read_isotherm(path: Path) -> dict[float, tuple[float, float]]:
    """Pressure -> (free width, excess wt%) for one isotherm folder."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    porosity = data["porosidad"]
    width = porosity["largest_free_sphere"]
    return {p["presion_mpa"]: (width, excess_weight_percent(p, porosity))
            for p in data["puntos"]}


def main() -> int:
    """Draw the three panels and write the figure beside the data."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE

    slits = {}
    for path in sorted(root.glob("rendija_*.json"),
                       key=lambda p: float(p.stem.split("_")[1])):
        got = read_isotherm(path)
        if got:
            slits[float(path.stem.split("_")[1])] = got
    if not slits:   # a working folder, where each width is an isotherm folder
        for folder in sorted(root.glob("iso_*"),
                             key=lambda p: float(p.name.split("_")[1])):
            got = read_isotherm(folder / "resultados.json")
            if got:
                slits[float(folder.name.split("_")[1])] = got

    schwarzites = {
        name: read_isotherm(SCHWARZITES / f"{f}.json")
        for name, f in (("Schwarz D", "diamond"), ("Schwarz P", "primitive"),
                        ("giroide", "gyroid"))}
    supers = {
        name: read_isotherm(SUPERS / f"{f}.json")
        for name, f in (("súper-grafeno", "supergrafeno"),
                        ("superfulereno", "superfulereno"),
                        ("supertubo", "supertubo"))}
    # The two shrunken super-graphenes belong to the same family and
    # ride the same colour. They are the point of that series now: they
    # sit at the slit's best widths and still fall under its curve.
    for label, name in (("súper-grafeno e22", "supergrafeno_e22_r4"),
                        ("súper-grafeno e26", "supergrafeno_e26_r5")):
        got = read_isotherm(HERE / f"{name}.json")
        if got:
            supers[label] = got

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")

    for axis, pressure in zip(axes, PRESSURES):
        axis.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
        axis.set_axisbelow(True)
        axis.axhline(0.0, color=MUTED, linewidth=1.0, zorder=1)
        for layers in LAYERS:
            axis.axvline(layers * SIGMA_H2, color=MUTED, linewidth=0.9,
                         linestyle=(0, (2, 3)), zorder=2)
        axis.set_xscale("log")
        axis.set_xticks(X_TICKS)
        axis.set_xticklabels([str(t) for t in X_TICKS])
        axis.minorticks_off()
        axis.tick_params(colors=INK2, labelsize=9)

        series = [(width, value) for width, value in
                  sorted((v[pressure] for v in slits.values()
                          if pressure in v))]
        if series:
            axis.plot([w for w, _ in series], [v for _, v in series],
                      color=SLIT, linewidth=2.0, marker="o", markersize=5.5,
                      markeredgecolor="#fcfcfb", markeredgewidth=1.2, zorder=4)
            peak = max(series, key=lambda s: s[1])
            axis.annotate(f"{peak[1]:.2f}", xy=peak, xytext=(0, 11),
                          textcoords="offset points", ha="center",
                          color=SLIT, fontsize=9, fontweight="bold", zorder=6)

        for group, colour in ((schwarzites, SCHWARZITE), (supers, SUPER)):
            points = [v[pressure] for v in group.values() if pressure in v]
            if points:
                axis.scatter([w for w, _ in points], [v for _, v in points],
                             s=60, color=colour, edgecolor="#fcfcfb",
                             linewidth=1.2, zorder=5)

        axis.set_title(f"{pressure:g} MPa", color=INK, fontsize=11,
                       fontweight="bold", loc="left")
        axis.set_xlabel("ancho de canal libre (Å)", color=INK2, fontsize=9)

    axes[0].set_ylabel("exceso adsorbido (wt %)", color=INK2, fontsize=9)

    # The layer marks are labelled once, in the first panel, and LOW:
    # the top of that panel holds the title and the peak's value, and
    # the first version put all three on top of each other.
    for layers in LAYERS:
        axes[0].annotate(f"{layers}σ",
                         xy=(layers * SIGMA_H2, 0.06),
                         xycoords=("data", "axes fraction"),
                         xytext=(0, 0), textcoords="offset points",
                         ha="center", va="bottom", color=MUTED, fontsize=8)

    # Direct labels, in the first panel only: three of them in every
    # panel would be clutter, and the identity carries across.
    first = slits.get(min(slits)) if slits else None
    if first and 1.0 in first:
        # Low and left, in the empty corner: the band between 3 and
        # 11 A now carries the slit's own dip, the schwarzite label and
        # two shrunken super-graphenes.
        axes[0].annotate("rendija de grafeno", xy=first[1.0],
                         xytext=(0, -48), textcoords="offset points",
                         color=SLIT, fontsize=9, fontweight="bold")
    if 1.0 in schwarzites["Schwarz D"]:
        # Below its dot, not beside it: the shrunken super-graphenes
        # landed at 9-10 A and the label used to run straight over them.
        axes[0].annotate("schwarzitas", xy=schwarzites["Schwarz D"][1.0],
                         xytext=(0, -20), textcoords="offset points",
                         ha="center", color=SCHWARZITE, fontsize=9,
                         fontweight="bold")
    if 1.0 in supers["supertubo"]:
        axes[0].annotate("superestructuras", xy=supers["supertubo"][1.0],
                         xytext=(-10, -20), textcoords="offset points",
                         color=SUPER, fontsize=9, fontweight="bold",
                         ha="right")

    handles = [
        Line2D([], [], color=SLIT, linewidth=2.0, marker="o", markersize=5.5,
               markeredgecolor=SURFACE, label="rendija de grafeno"),
        Line2D([], [], color=SCHWARZITE, linestyle="none", marker="o",
               markersize=7, markeredgecolor=SURFACE, label="schwarzitas"),
        Line2D([], [], color=SUPER, linestyle="none", marker="o",
               markersize=7, markeredgecolor=SURFACE, label="superestructuras"),
    ]
    legend = fig.legend(handles=handles, loc="upper right", ncols=3,
                        frameon=False, fontsize=9,
                        bbox_to_anchor=(0.995, 1.005))
    for text in legend.get_texts():
        text.set_color(INK2)

    fig.suptitle("Hidrógeno a 77 K: lo que decide la captación es el ancho "
                 "del poro", color=INK, fontsize=13, fontweight="bold",
                 x=0.012, ha="left", y=0.98)
    fig.text(0.012, 0.045,
             "La rendija es un barrido controlado: su área es constante "
             "(2670 m²/g) en toda la serie, así que sólo se mueve el ancho.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.text(0.012, 0.012,
             "Las marcas nσ son múltiplos del diámetro del H₂ (2.958 Å), "
             "es decir el ancho mínimo que admite n capas de hidrógeno.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"ancho_vs_exceso.{suffix}", dpi=200,
                    facecolor=fig.get_facecolor())
    print(f"escrito {root / 'ancho_vs_exceso.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
