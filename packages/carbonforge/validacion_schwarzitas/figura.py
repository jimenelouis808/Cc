"""Uptake against surface area: why these numbers differ from Kowalczyk's.

One series of four structures, so there is no legend box — the title
names what the dots are and each dot is labelled. The reference line is
the Chahine rule (about 1 wt% per 500 m2/g at 77 K), drawn in muted ink
because it is context and not data.

The figure exists to answer one question and should not be asked to
answer two: is the gap to Kowalczyk's ~3 wt% a difference of method or
of material? If the dots sit on the line and his value sits at the area
that line assigns it, the answer is material.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, INK, INK2, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CHAHINE = 500.0
#: m2/g from `analyse_porosity`: geometric, not BET.
AREA = {"haz SWNT": 2590, "primitive": 2543, "diamond": 1717, "gyroid": 2534}
#: The bundle's own 3.8 MPa value is INTERPOLATED between the validated
#: 77 K isotherm's 3.0 MPa (4.43 wt%) and 5.0 MPa (4.58) points -- the
#: isotherm is flat there, so the interpolation is safe, but it is not a
#: measurement and the figure labels it so. Putting an unmarked
#: interpolation beside three measured points would be the quiet kind of
#: wrong this whole exercise is about avoiding.
BUNDLE_AT_38 = 4.49


def load(base: pathlib.Path) -> dict[str, float]:
    got = {"haz SWNT": BUNDLE_AT_38}
    for kind in ("primitive", "diamond", "gyroid"):
        path = base / f"iso_{kind}" / "resultados.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text())["puntos"]:
            if abs(row["presion_mpa"] - 3.8) < 1e-6:
                got[kind] = row["wt_pct_total"]
    return got


#: Where each label goes, in points from its dot. Hand-placed because
#: four points at two clustered areas collide under any rule simple
#: enough to state: the first version put "primitive" on top of its own
#: marker and "gyroid" on top of that.
LABEL_OFFSET = {
    "haz SWNT": (0, -34),
    "primitive": (-62, -6),
    "diamond": (0, 14),
    "gyroid": (0, 14),
}


def draw(values: dict[str, float], out: pathlib.Path) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 4.8))
    axis.set_facecolor("#fcfcfb")
    axis.grid(True, color=GRID, lw=0.7, zorder=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(colors=MUTED, labelcolor=INK2, width=0.8)

    span = np.array([1300.0, 2850.0])
    axis.plot(span, span / CHAHINE, "--", color=MUTED, lw=1.3, zorder=1)
    axis.annotate("regla de Chahine:\n1 wt% por 500 m²/g",
                  xy=(1340, 1340 / CHAHINE + 0.12), color=MUTED, fontsize=8,
                  ha="left", va="bottom")

    axis.axhline(3.0, color=GRID, lw=7, zorder=0)
    axis.annotate("Kowalczyk: ~3 wt% para P, D y G, leído de una figura",
                  xy=(2830, 2.72), color=MUTED, fontsize=8, ha="right",
                  va="center")

    # The pair that stops this being a one-variable story: nearly the
    # same area, 42 % apart. Drawn, because a reader who only sees the
    # Chahine line would take it for a law.
    pair = [("haz SWNT", values["haz SWNT"]), ("gyroid", values["gyroid"])]
    axis.annotate(
        "", xy=(AREA["gyroid"], pair[1][1]), xytext=(AREA["haz SWNT"],
                                                     pair[0][1]),
        arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1,
                        shrinkA=9, shrinkB=9), zorder=2)
    axis.annotate("misma área,\n+42 %", xy=(2600, 5.45), color=MUTED,
                  fontsize=8, ha="left", va="center")

    for name, wt in values.items():
        axis.plot([AREA[name]], [wt], "o", color=BLUE, ms=11,
                  mec="#fcfcfb", mew=1.8, zorder=4)
        mark = "\n(interpolado)" if name == "haz SWNT" else ""
        offset = LABEL_OFFSET[name]
        axis.annotate(f"{name}  {wt:.2f} wt%{mark}", xy=(AREA[name], wt),
                      xytext=offset, textcoords="offset points",
                      ha="center" if offset[0] == 0 else "right",
                      va="center", color=INK2, fontsize=8.5, zorder=5)

    axis.set_xlabel("área accesible geométrica  (m² g⁻¹)", color=INK2)
    axis.set_ylabel("H₂ absoluto a 3.8 MPa y 77 K  (wt%, masa total)",
                    color=INK2)
    axis.set_title("Captación contra área accesible: el área ordena, "
                   "y no basta",
                   color=INK, fontsize=11, loc="left", pad=10)
    axis.set_xlim(1300, 2850)
    axis.set_ylim(2.4, 7.3)
    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight",
                   facecolor="#fcfcfb")
    figure.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight",
                   facecolor="#fcfcfb")
    plt.close(figure)


def main() -> int:
    base = pathlib.Path(sys.argv[1])
    out_dir = pathlib.Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    values = load(base)
    draw(values, out_dir / "capacidad_vs_area")
    for name, wt in values.items():
        print(f"{name:<11} {AREA[name]:5d} m2/g  {wt:5.2f} wt%  "
              f"razon {wt / (AREA[name] / CHAHINE):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
