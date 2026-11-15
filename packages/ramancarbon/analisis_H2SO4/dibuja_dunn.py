"""Draw the Dunn separation from the exported CSVs, with no analysis.

    uv run python analisis_H2SO4/dibuja_dunn.py [velocidad_mV_s]

The point of this script is that it does NOT import ramancarbon. It
reads `dunn_lazo_<nu>mVs.csv`, which holds the cycle as one ordered
path, and draws it in three calls. If you want the figure in another
tool, this is the shape the data have to be in and these are the three
calls to translate.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

#: Enough contrast in both themes and in grey-scale, and the same two
#: the package's own panel uses.
CAPACITIVE = "#2a78d6"
DIFFUSIVE = "#eb6834"


def read(path: Path) -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                if key in ("rama", "separacion_definida"):
                    continue
                columns.setdefault(key, []).append(
                    float(value) if value else float("nan"))
    return columns


def main(rate: str = "5") -> None:
    path = HERE / f"dunn_lazo_{rate}mVs.csv"
    if not path.exists():
        found = sorted(p.name for p in HERE.glob("dunn_lazo_*mVs.csv"))
        raise SystemExit(f"no está {path.name}. Hay: {', '.join(found)}")
    data = read(path)
    per_gram = "i_A_g" in data
    unit = "A g$^{-1}$" if per_gram else "mA"
    potential = data["E_V"]
    total = data["i_A_g" if per_gram else "i_mA"]
    capacitive = data["i_cap_A_g" if per_gram else "i_cap_mA"]

    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    # Three calls. No branch logic: the file IS the path.
    axis.fill_between(potential, capacitive, total, color=DIFFUSIVE,
                      alpha=0.85, linewidth=0, label="difusivo ($k_2\\sqrt{\\nu}$)")
    axis.fill_between(potential, 0.0, capacitive, color=CAPACITIVE,
                      alpha=0.85, linewidth=0, label="capacitivo ($k_1\\nu$)")
    axis.plot(potential, total, color="#1a1a19", linewidth=1.3,
              label=f"{rate} mV/s")

    axis.axhline(0.0, color="#8a8a84", linewidth=0.6, zorder=0)
    axis.set_xlabel("E / V vs Ag/AgCl (KCl sat.)")
    axis.set_ylabel(f"i / {unit}")
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    axis.tick_params(which="both", direction="in", top=True, right=True)
    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(HERE / f"dunn_{rate}mVs{suffix}", dpi=600)
    print(f"escrita dunn_{rate}mVs.{{png,pdf}}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "5")
