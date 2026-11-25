"""Figure of tamizado.txt: H2/D2 selectivity at 77 and 40 K, valid values
filled and upper bounds hollow (criterion 2*dq < 10 % of q_st(H2)).

    python tamizado_figura.py   -> figuras/tamizado.png
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROW = re.compile(r"^(.*?)\s*\|\s*(\d+) K\s*\|.*?S\s+([\d.]+).*?q_st H2\s+([\d.]+) D2\s+([\d.]+)")
SHOW = ["slit H=6.2", "slit H=6.4", "slit H=6.6", "slit H=7", "slit H=8",
        "SWCNT bundle gap=4", "SWCNT bundle gap=6", "super-hypercube", "super-cube",
        "super-tetrahedron", "super-fcc", "super-cubic", "super-diamond",
        "schwarzite-P", "schwarzite-D", "gyroid", "knee-super-graphene", "knee-super-square"]
LABEL = {"slit H=6.2": "slit 6.2 Å", "slit H=6.4": "slit 6.4 Å", "slit H=6.6": "slit 6.6 Å",
         "slit H=7": "slit 7 Å", "slit H=8": "slit 8 Å",
         "SWCNT bundle gap=4": "SWCNT bundle, 4 Å gap", "SWCNT bundle gap=6": "SWCNT bundle, 6 Å gap",
         "knee-super-graphene": "super-graphene (cryst.)", "knee-super-square": "super-square (cryst.)",
         "schwarzite-P": "Schwarz P", "schwarzite-D": "Schwarz D"}


def parse():
    out = {}
    for line in (HERE / "tamizado.txt").read_text().splitlines():
        m = ROW.match(line)
        if m:
            name, t, s, qh, qd = m.groups()
            valid = 2 * (float(qd) - float(qh)) < 0.1 * float(qh)
            out[(name.strip(), int(t))] = (float(s), valid)
    return out


def main() -> None:
    d = parse()
    names = [n for n in SHOW if (n, 77) in d]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True)
    for ax, t, colour in ((axes[0], 77, "#0f6b74"), (axes[1], 40, "#b5651d")):
        for i, n in enumerate(names):
            s, ok = d[(n, t)]
            ax.barh(i, s - 1, left=1, color=colour if ok else "white",
                    edgecolor=colour, hatch=None if ok else "///", height=0.7)
            ax.text(s + 0.03, i, f"{s:.2f}" + ("" if ok else " (bound)"), va="center",
                    fontsize=8, color="#1c2126")
        ax.axvline(1, color="#5d6771", lw=0.8)
        ax.set_title(f"{t} K", loc="left", fontsize=11)
        ax.set_xlabel("S(D₂/H₂) = K_H(D₂) / K_H(H₂)")
        ax.set_xlim(1, (5.5 if t == 40 else 2.0))
    axes[0].set_yticks(range(len(names)), [LABEL.get(n, n) for n in names], fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Zero-coverage H₂/D₂ quantum sieving (Feynman-Hibbs); hatched = upper bound",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    (HERE / "figuras").mkdir(exist_ok=True)
    fig.savefig(HERE / "figuras" / "tamizado.png", dpi=170)


if __name__ == "__main__":
    main()
